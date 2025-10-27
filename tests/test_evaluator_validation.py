"""Tests for evaluator validation logic."""

import datetime
from unittest.mock import Mock

import pytest
from returns.result import Failure, Success

from dqx.common import ResultKey, SymbolicValidator
from dqx.evaluator import Evaluator
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph
from dqx.provider import MetricProvider


class TestEvaluatorValidation:
    """Test suite for evaluator validation functionality."""

    def test_validator_execution_success(self) -> None:
        """Test that validators are properly executed and return OK for passing assertions."""
        # Setup
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Create a metric that returns 100
        x1 = provider.average("price", dataset="orders")
        provider.index[x1].fn = lambda k: Success(100.0)

        # Create graph with assertion that should pass
        root = RootNode("test_suite")
        check = root.add_check("price_check", datasets=["orders"])

        # Add assertion with validator that checks > 50
        validator = SymbolicValidator("> 50", lambda x: x > 50)
        assertion = check.add_assertion(x1, name="price > 50", validator=validator)

        # Execute
        evaluator = Evaluator(provider, key, "Test Suite")
        evaluator.visit(assertion)

        # Verify
        assert assertion._metric is not None
        assert isinstance(assertion._metric, Success)
        assert assertion._metric.unwrap() == 100.0
        assert assertion._result == "OK"

    def test_validator_execution_failure(self) -> None:
        """Test that validators return FAILURE for failing assertions."""
        # Setup
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Create a metric that returns 10
        x1 = provider.average("price", dataset="orders")
        provider.index[x1].fn = lambda k: Success(10.0)

        # Create graph with assertion that should fail
        root = RootNode("test_suite")
        check = root.add_check("price_check", datasets=["orders"])

        # Add assertion with validator that checks > 50
        validator = SymbolicValidator("> 50", lambda x: x > 50)
        assertion = check.add_assertion(x1, name="price > 50", validator=validator)

        # Execute
        evaluator = Evaluator(provider, key, "Test Suite")
        evaluator.visit(assertion)

        # Verify
        assert assertion._metric is not None
        assert isinstance(assertion._metric, Success)
        assert assertion._metric.unwrap() == 10.0
        assert assertion._result == "FAILURE"

    def test_metric_failure_results_in_failure_status(self) -> None:
        """Test that metric computation failures result in FAILURE status."""
        # Setup
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Create a metric that fails
        x1 = provider.average("price", dataset="orders")
        provider.index[x1].fn = lambda k: Failure("Database error")

        # Create graph with assertion
        root = RootNode("test_suite")
        check = root.add_check("price_check", datasets=["orders"])

        # Add assertion with validator
        validator = SymbolicValidator("> 50", lambda x: x > 50)
        assertion = check.add_assertion(x1, name="price > 50", validator=validator)

        # Execute
        evaluator = Evaluator(provider, key, "Test Suite")
        evaluator.visit(assertion)

        # Verify
        assert assertion._metric is not None
        assert isinstance(assertion._metric, Failure)
        assert assertion._result == "FAILURE"

    def test_validator_exception_handling(self) -> None:
        """Test that validator exceptions are properly handled."""
        # Setup
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Create a metric
        x1 = provider.average("price", dataset="orders")
        provider.index[x1].fn = lambda k: Success(100.0)

        # Create graph with assertion that has a failing validator
        root = RootNode("test_suite")
        check = root.add_check("price_check", datasets=["orders"])

        # Add assertion with validator that raises exception
        def failing_validator(x: float) -> bool:
            raise ValueError("Validator error")

        validator = SymbolicValidator("failing", failing_validator)
        assertion = check.add_assertion(x1, name="will fail", validator=validator)

        # Execute - should raise DQXError
        evaluator = Evaluator(provider, key, "Test Suite")
        with pytest.raises(Exception) as exc_info:
            evaluator.visit(assertion)

        assert "Validator execution failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_mixed_validation_results(self) -> None:
        """Test a graph with mixed validation results."""
        # Setup
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Create metrics
        x1 = provider.average("price", dataset="orders")
        provider.index[x1].fn = lambda k: Success(100.0)

        x2 = provider.minimum("quantity", dataset="inventory")
        provider.index[x2].fn = lambda k: Success(5.0)

        x3 = provider.maximum("discount", dataset="orders")
        provider.index[x3].fn = lambda k: Success(0.8)

        # Create graph
        root = RootNode("test_suite")
        check1 = root.add_check("price_checks", datasets=["orders"])
        check2 = root.add_check("inventory_checks", datasets=["inventory"])

        # Add assertions with different outcomes
        assertion1 = check1.add_assertion(
            x1, name="price > 50", validator=SymbolicValidator("> 50", lambda x: x > 50)
        )  # Should pass: 100 > 50

        assertion2 = check1.add_assertion(
            x3, name="discount < 0.5", validator=SymbolicValidator("< 0.5", lambda x: x < 0.5)
        )  # Should fail: 0.8 is not < 0.5

        assertion3 = check2.add_assertion(
            x2, name="quantity >= 5", validator=SymbolicValidator("≥ 5", lambda x: x >= 5)
        )  # Should pass: 5 >= 5

        # Execute
        evaluator = Evaluator(provider, key, "Test Suite")
        graph = Graph(root)
        await graph.async_dfs(evaluator)

        # Verify results
        assert assertion1._result == "OK"
        assert assertion2._result == "FAILURE"
        assert assertion3._result == "OK"

    def test_complex_expression_validation(self) -> None:
        """Test validation of complex expressions."""
        # Setup
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Create metrics
        x1 = provider.average("price", dataset="orders")
        provider.index[x1].fn = lambda k: Success(100.0)

        x2 = provider.average("cost", dataset="orders")
        provider.index[x2].fn = lambda k: Success(60.0)

        # Create graph with complex expression
        root = RootNode("test_suite")
        check = root.add_check("margin_check", datasets=["orders"])

        # Profit margin: (price - cost) / price > 0.3
        margin_expr = (x1 - x2) / x1
        validator = SymbolicValidator("> 0.3", lambda x: x > 0.3)
        assertion = check.add_assertion(margin_expr, name="profit margin > 30%", validator=validator)

        # Execute
        evaluator = Evaluator(provider, key, "Test Suite")
        evaluator.visit(assertion)

        # Verify
        assert assertion._metric is not None
        assert isinstance(assertion._metric, Success)
        # (100 - 60) / 100 = 0.4
        assert abs(assertion._metric.unwrap() - 0.4) < 0.0001
        assert assertion._result == "OK"  # 0.4 > 0.3

    def test_edge_case_validation(self) -> None:
        """Test validation with edge cases like zero, negative, infinity."""
        # Setup
        provider = MetricProvider(db=Mock())
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Create all metrics upfront (before evaluator is created)
        x1 = provider.sum("zero_col", dataset="data")
        provider.index[x1].fn = lambda k: Success(0.0)

        x2 = provider.sum("negative_col", dataset="data")
        provider.index[x2].fn = lambda k: Success(-10.0)

        root = RootNode("test_suite")
        check = root.add_check("edge_cases", datasets=["data"])

        # Test that zero is handled correctly
        assertion1 = check.add_assertion(
            x1, name="zero is not positive", validator=SymbolicValidator("> 0", lambda x: x > 0)
        )

        evaluator = Evaluator(provider, key, "Test Suite")
        evaluator.visit(assertion1)

        assert assertion1._result == "FAILURE"  # 0 is not > 0

        # Test negative value
        assertion2 = check.add_assertion(x2, name="negative check", validator=SymbolicValidator("< 0", lambda x: x < 0))

        evaluator.visit(assertion2)
        assert assertion2._result == "OK"  # -10 < 0


class TestValidationExpressions:
    """Test that validation expressions are properly constructed."""

    def test_expression_includes_validator_name(self) -> None:
        """Test that collected results include full validation expressions."""
        from dqx.api import Context, VerificationSuite, check

        @check(name="test check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("price")).where(name="Price validation").is_gt(50)

        # Setup

        db = Mock()

        # Mock the database session and query
        mock_session = Mock()
        mock_query = Mock()
        mock_query.all.return_value = []  # Return empty list for metrics
        mock_session.query.return_value = mock_query
        db.new_session.return_value = mock_session

        suite = VerificationSuite([test_check], db, "Test Suite")

        # Mock the provider's metrics
        for symbol_ref, symbol_info in suite.provider.index.items():
            # Create a mock that returns a proper float value
            symbol_info.fn = lambda k: Success(75.0)

        # Run suite which will build graph internally
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        import pyarrow as pa

        from dqx.datasource import DuckRelationDataSource

        data = pa.table({"price": [75.0]})

        # Disable plugins to avoid metric trace issues
        suite.run([DuckRelationDataSource.from_arrow(data, "data")], key, enable_plugins=False)

        # Get the assertion node
        assertions = list(suite.graph.assertions())
        assert len(assertions) == 1

        assertion = assertions[0]

        # Verify validator name is set correctly
        assert assertion.validator is not None
        assert assertion.validator.name == "> 50"

        # Test expression construction in collect_results
        # Set up mock values
        assertion._metric = Success(75.0)
        assertion._result = "OK"

        results = suite.collect_results()

        # Verify expression includes validator
        assert len(results) == 1
        result = results[0]

        # Expression should be something like "x_1 > 50"
        assert result.expression is not None
        assert ">" in result.expression
        assert "50" in result.expression
