"""Integration tests for the Evaluator with EvaluationFailure."""

from datetime import date
from unittest.mock import Mock

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx.common import EvaluationFailure, ResultKey, SymbolicValidator
from dqx.evaluator import Evaluator
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph
from dqx.provider import MetricProvider


class TestEvaluatorIntegration:
    """Integration tests for Evaluator with the full graph traversal."""

    @pytest.mark.asyncio
    async def test_full_graph_traversal_with_failures(self) -> None:
        """Test complete graph traversal with metric failures."""
        # Create provider with some failing metrics
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})

        # Add successful metrics
        x1 = provider.average("price", dataset="orders")
        # Mock the metric to return success
        provider.index[x1].fn = lambda k: Success(100.0)

        x2 = provider.sum("quantity", dataset="inventory")
        # Mock the metric to return success
        provider.index[x2].fn = lambda k: Success(50.0)

        # Add failing metric
        x3 = provider.average("cost", dataset="orders")
        # Mock the metric to return failure
        provider.index[x3].fn = lambda k: Failure("Database connection error")

        # Create graph structure
        root = RootNode("test_suite")
        check1 = root.add_check("price_checks", datasets=["orders"])
        check2 = root.add_check("inventory_checks", datasets=["inventory"])

        # Add assertions
        price_validator = SymbolicValidator("> 0", lambda x: x > 0)
        combined_validator = SymbolicValidator("> 0", lambda x: x > 0)
        quantity_validator = SymbolicValidator(">= 0", lambda x: x >= 0)

        assertion1 = check1.add_assertion(x1, name="avg_price", validator=price_validator)
        assertion2 = check1.add_assertion(x1 + x3, name="avg_price + avg_cost", validator=combined_validator)
        assertion3 = check2.add_assertion(x2, name="sum_quantity", validator=quantity_validator)

        # Create evaluator and traverse
        evaluator = Evaluator(provider, key, "Test Suite")
        graph = Graph(root)
        await graph.async_dfs(evaluator)

        # Check results
        # Assertion 1: Should succeed with value 100.0
        assert isinstance(assertion1._metric, Success)
        assert assertion1._metric.unwrap() == 100.0

        # Assertion 2: Should fail due to x3 failure
        assert isinstance(assertion2._metric, Failure)
        failures = assertion2._metric.failure()
        assert len(failures) == 1
        failure = failures[0]
        assert isinstance(failure, EvaluationFailure)
        assert failure.error_message == "One or more metrics failed to evaluate"
        assert str(x1 + x3) in failure.expression
        assert len(failure.symbols) == 2

        # Find the failed symbol by checking the metric name in the symbol info
        failed_symbol = next(s for s in failure.symbols if "cost" in s.metric)
        assert isinstance(failed_symbol.value, Failure)
        assert failed_symbol.value.failure() == "Database connection error"

        # Assertion 3: Should succeed with value 50.0
        assert isinstance(assertion3._metric, Success)
        assert assertion3._metric.unwrap() == 50.0

    @pytest.mark.asyncio
    async def test_nan_handling_in_graph_traversal(self) -> None:
        """Test NaN handling during graph traversal."""
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})

        # Create metrics that will produce NaN
        x1 = provider.average("col1", dataset="dataset1")
        provider.index[x1].fn = lambda k: Success(0.0)

        x2 = provider.sum("col2", dataset="dataset1")
        provider.index[x2].fn = lambda k: Success(0.0)

        # Create graph with assertion that produces NaN
        root = RootNode("nan_test_suite")
        check = root.add_check("nan_check", datasets=["dataset1"])
        # Use sp.zoo to force NaN result
        nan_validator = SymbolicValidator("finite", lambda x: not float("inf") and not float("-inf"))
        assertion = check.add_assertion(sp.zoo * x1 / x2, name="sp.zoo * x1 / x2", validator=nan_validator)

        # Traverse graph
        evaluator = Evaluator(provider, key, "Test Suite")
        graph = Graph(root)
        await graph.async_dfs(evaluator)

        # Check NaN is handled properly
        assert isinstance(assertion._metric, Failure)
        failures = assertion._metric.failure()
        assert len(failures) == 1
        failure = failures[0]
        assert failure.error_message == "Validating value is NaN"
        assert len(failure.symbols) == 2

    def test_evaluator_with_multiple_datasets(self) -> None:
        """Test evaluator with metrics from multiple datasets."""
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={"env": "prod"})

        # Metrics from different datasets
        orders_metric = provider.sum("amount", dataset="orders")
        provider.index[orders_metric].fn = lambda k: Success(1000.0)

        inventory_metric = provider.sum("quantity", dataset="inventory")
        provider.index[inventory_metric].fn = lambda k: Success(500.0)

        users_metric = provider.sum("active", dataset="users")
        provider.index[users_metric].fn = lambda k: Success(100.0)

        # Complex expression across datasets
        expr = (orders_metric / users_metric) - (inventory_metric / users_metric)

        evaluator = Evaluator(provider, key, "Test Suite")
        result = evaluator.evaluate(expr)

        # Should succeed: (1000/100) - (500/100) = 10 - 5 = 5
        assert isinstance(result, Success)
        assert result.unwrap() == 5.0

    def test_symbol_info_completeness(self) -> None:
        """Test that SymbolInfo contains all necessary information."""
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=date.today(), tags={})

        # Create a failing metric with detailed info
        symbol = provider.average("revenue", dataset="transactions")
        provider.index[symbol].fn = lambda k: Failure("Query timeout after 30s")

        evaluator = Evaluator(provider, key, "Test Suite")
        result = evaluator.evaluate(symbol * 2)

        assert isinstance(result, Failure)
        failures = result.failure()
        failure = failures[0]

        # Check symbol info
        symbol_info = failure.symbols[0]
        assert symbol_info.name == str(symbol)  # Will be x_1 or similar
        assert symbol_info.metric == "average(revenue)"
        assert symbol_info.dataset == "transactions"
        assert isinstance(symbol_info.value, Failure)
        assert "timeout" in symbol_info.value.failure().lower()
