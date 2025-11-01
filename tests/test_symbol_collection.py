"""Tests for the symbol collection feature."""

from datetime import date
from typing import Any
from unittest.mock import Mock

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx.common import DQXError, ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider, SymbolicMetric, SymbolInfo


class TestSymbolInfoExtended:
    """Tests for the extended SymbolInfo dataclass."""

    def test_symbol_info_with_all_fields(self) -> None:
        """Test creating SymbolInfo with all new fields."""
        info = SymbolInfo(
            name="x_1",
            metric="average(price)",
            dataset="orders",
            value=Success(100.5),
            yyyy_mm_dd=date(2025, 1, 13),
            tags={"env": "prod", "region": "us-east"},
        )

        assert info.name == "x_1"
        assert info.metric == "average(price)"
        assert info.dataset == "orders"
        assert info.value == Success(100.5)
        assert info.yyyy_mm_dd == date(2025, 1, 13)
        assert info.tags == {"env": "prod", "region": "us-east"}

    def test_symbol_info_with_empty_tags(self) -> None:
        """Test that tags defaults to empty dict."""
        info = SymbolInfo(
            name="x_1",
            metric="average(price)",
            dataset="orders",
            value=Success(100.5),
            yyyy_mm_dd=date(2025, 1, 13),
            # tags not provided, should default to {}
        )

        assert info.tags == {}

    def test_symbol_info_with_none_dataset(self) -> None:
        """Test SymbolInfo with None dataset."""
        info = SymbolInfo(
            name="x_1", metric="constant", dataset=None, value=Success(42.0), yyyy_mm_dd=date(2025, 1, 13)
        )

        assert info.dataset is None


class TestGetSymbolWithString:
    """Tests for get_symbol method with string input."""

    def test_get_symbol_with_string(self) -> None:
        """Test get_symbol accepts string and returns correct SymbolicMetric."""
        provider = Mock(spec=MetricProvider)
        symbol = sp.Symbol("x_1")
        metric_spec = Mock(__str__=Mock(return_value="average(price)"))

        symbolic_metric = SymbolicMetric(
            name="x_1", symbol=symbol, fn=lambda k: Success(100.0), metric_spec=metric_spec, lag=0, dataset="orders"
        )

        provider._metrics = [symbolic_metric]
        provider.get_symbol.return_value = symbolic_metric

        # Test with string
        result = provider.get_symbol("x_1")
        assert result == symbolic_metric
        provider.get_symbol.assert_called_with("x_1")

    def test_get_symbol_with_symbol(self) -> None:
        """Test get_symbol still works with Symbol objects."""
        provider = Mock(spec=MetricProvider)
        symbol = sp.Symbol("x_1")
        metric_spec = Mock(__str__=Mock(return_value="average(price)"))

        symbolic_metric = SymbolicMetric(
            name="x_1", symbol=symbol, fn=lambda k: Success(100.0), metric_spec=metric_spec, lag=0, dataset="orders"
        )

        provider._metrics = [symbolic_metric]
        provider.get_symbol.return_value = symbolic_metric

        # Test with Symbol
        result = provider.get_symbol(symbol)
        assert result == symbolic_metric
        provider.get_symbol.assert_called_with(symbol)

    def test_get_symbol_string_not_found(self) -> None:
        """Test get_symbol raises error for non-existent string symbol."""
        from dqx.provider import SymbolicMetricBase

        provider = SymbolicMetricBase()

        with pytest.raises(DQXError, match="Symbol x_99 not found"):
            provider.get_symbol("x_99")


class TestCollectSymbols:
    """Tests for the collect_symbols method."""

    def test_collect_symbols_simple_expression(self) -> None:
        """Test collecting symbols from a simple expression."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date(2025, 1, 13), tags={"env": "test"})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Create symbols
        x1 = sp.Symbol("x_1")
        x2 = sp.Symbol("x_2")

        # Create symbolic metrics
        metric_spec1 = Mock(__str__=Mock(return_value="average(price)"))
        metric_spec2 = Mock(__str__=Mock(return_value="sum(quantity)"))

        sm1 = SymbolicMetric(
            name="average(price)",  # This is what gets stored in SymbolInfo.metric
            symbol=x1,
            fn=lambda k: Success(100.0),
            metric_spec=metric_spec1,
            lag=0,
            dataset="orders",
        )
        sm2 = SymbolicMetric(
            name="sum(quantity)",  # This is what gets stored in SymbolInfo.metric
            symbol=x2,
            fn=lambda k: Success(50.0),
            metric_spec=metric_spec2,
            lag=0,
            dataset="inventory",
        )

        provider.get_symbol.side_effect = lambda s: sm1 if s == x1 else sm2
        evaluator._metrics = {x1: Success(100.0), x2: Success(50.0)}

        # Collect symbols
        expr = x1 + x2
        symbols = evaluator.collect_symbols(expr)

        assert len(symbols) == 2

        # Check symbol info
        for si in symbols:
            if si.name == "x_1":
                assert si.metric == "average(price)"
                assert si.dataset == "orders"
                assert si.value == Success(100.0)
            else:
                assert si.name == "x_2"
                assert si.metric == "sum(quantity)"
                assert si.dataset == "inventory"
                assert si.value == Success(50.0)

            # Check context fields
            assert si.yyyy_mm_dd == date(2025, 1, 13)
            assert si.tags == {"env": "test"}

    def test_collect_symbols_with_failures(self) -> None:
        """Test collecting symbols when some metrics fail."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date(2025, 1, 13), tags={})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Create symbols
        x1 = sp.Symbol("x_1")
        x2 = sp.Symbol("x_2")

        # Create symbolic metrics
        metric_spec1 = Mock(__str__=Mock(return_value="average(price)"))
        metric_spec2 = Mock(__str__=Mock(return_value="sum(quantity)"))

        sm1 = SymbolicMetric(
            name="x_1", symbol=x1, fn=lambda k: Success(100.0), metric_spec=metric_spec1, lag=0, dataset="orders"
        )
        sm2 = SymbolicMetric(
            name="x_2",
            symbol=x2,
            fn=lambda k: Failure("Database error"),
            metric_spec=metric_spec2,
            lag=0,
            dataset="inventory",
        )

        provider.get_symbol.side_effect = lambda s: sm1 if s == x1 else sm2
        evaluator._metrics = {x1: Success(100.0), x2: Failure("Database error")}

        # Collect symbols
        expr = x1 * x2
        symbols = evaluator.collect_symbols(expr)

        assert len(symbols) == 2

        # Check that both symbols are collected, including the failed one
        success_count = sum(1 for si in symbols if isinstance(si.value, Success))
        failure_count = sum(1 for si in symbols if isinstance(si.value, Failure))

        assert success_count == 1
        assert failure_count == 1

    def test_collect_symbols_complex_expression(self) -> None:
        """Test collecting symbols from complex nested expressions."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date(2025, 1, 13), tags={"region": "us"})
        evaluator = Evaluator(provider, key, "Complex Suite")

        # Create symbols
        x1, x2, x3 = sp.symbols("x_1 x_2 x_3")

        # Create symbolic metrics
        metrics_data = [
            (x1, "x_1", "avg(price)", "orders", 100.0),
            (x2, "x_2", "sum(quantity)", "orders", 200.0),
            (x3, "x_3", "avg(rating)", "reviews", 4.5),
        ]

        def make_fn(v: float) -> Any:
            return lambda k: Success(v)

        symbolic_metrics = {}
        for sym, name, metric_str, dataset, value in metrics_data:
            metric_spec = Mock(__str__=Mock(return_value=metric_str))
            sm = SymbolicMetric(
                name=name, symbol=sym, fn=make_fn(value), metric_spec=metric_spec, lag=0, dataset=dataset
            )
            symbolic_metrics[sym] = sm

        provider.get_symbol.side_effect = lambda s: symbolic_metrics[s]
        evaluator._metrics = {sym: Success(value) for sym, _, _, _, value in metrics_data}

        # Complex expression
        expr = (x1 + x2) / x3 + sp.sqrt(x1 * x2)
        symbols = evaluator.collect_symbols(expr)

        assert len(symbols) == 3

        # Verify all symbols collected
        symbol_names = {si.name for si in symbols}
        assert symbol_names == {"x_1", "x_2", "x_3"}

        # Verify datasets
        datasets = {si.dataset for si in symbols}
        assert datasets == {"orders", "reviews"}

    def test_collect_symbols_empty_expression(self) -> None:
        """Test collecting symbols from constant expression."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date(2025, 1, 13), tags={})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Constant expression
        expr = sp.sympify(42)
        symbols = evaluator.collect_symbols(expr)

        assert len(symbols) == 0

    def test_collect_symbols_preserves_order(self) -> None:
        """Test that symbol collection preserves consistent order."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date(2025, 1, 13), tags={})
        evaluator = Evaluator(provider, key, "Test Suite")

        # Create multiple symbols
        symbols = [sp.Symbol(f"x_{i}") for i in range(1, 6)]

        # Set up mocks
        for i, sym in enumerate(symbols):
            metric_spec = Mock(__str__=Mock(return_value=f"metric_{i + 1}"))
            sm = SymbolicMetric(
                name=str(sym),
                symbol=sym,
                fn=lambda k: Success(float(i + 1)),
                metric_spec=metric_spec,
                lag=0,
                dataset=f"dataset_{i + 1}",
            )
            if i == 0:
                provider.get_symbol.side_effect = [sm] * len(symbols)

        evaluator._metrics = {sym: Success(float(i + 1)) for i, sym in enumerate(symbols)}

        # Expression using all symbols
        expr = sum(symbols)
        collected = evaluator.collect_symbols(expr)

        # Should have all symbols
        assert len(collected) == len(symbols)


class TestEvaluatorWithContext:
    """Test that Evaluator correctly populates context in SymbolInfo during evaluation."""

    def test_evaluator_populates_context_fields(self) -> None:
        """Test that _gather correctly populates all context fields."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date(2025, 1, 15), tags={"env": "staging", "version": "2.0"})
        evaluator = Evaluator(provider, key, "Production Suite")

        # Create a symbol
        x1 = sp.Symbol("x_1")
        metric_spec = Mock(__str__=Mock(return_value="average(revenue)"))

        sm = SymbolicMetric(
            name="x_1", symbol=x1, fn=lambda k: Success(1000.0), metric_spec=metric_spec, lag=0, dataset="sales"
        )

        provider.get_symbol.return_value = sm
        evaluator._metrics = {x1: Success(1000.0)}

        # Evaluate expression to trigger _gather
        result = evaluator.evaluate(x1 * 2)

        assert isinstance(result, Success)
        assert result.unwrap() == 2000.0

        # Now collect symbols to verify context
        symbols = evaluator.collect_symbols(x1)

        assert len(symbols) == 1
        si = symbols[0]

        # Verify all context fields
        assert si.yyyy_mm_dd == date(2025, 1, 15)
        assert si.tags == {"env": "staging", "version": "2.0"}
        assert si.dataset == "sales"
