"""Tests for the evaluator module."""

import datetime as dt

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx.common import DQXError, ExecutionId, ResultKey, SymbolicValidator
from dqx.evaluator import Evaluator
from dqx.graph.nodes import AssertionNode, CheckNode, RootNode
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider, SymbolicMetric


class TestEvaluatorBasics:
    """Test basic evaluator functionality."""

    def test_evaluator_initialization(self) -> None:
        """Test evaluator initialization."""
        # Create real provider with metrics
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={"env": "prod"})

        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        assert evaluator.provider is provider
        assert evaluator._key == key
        assert evaluator._suite_name == "Test Suite"
        assert evaluator._data_av_threshold == 0.8
        assert evaluator._metrics is None  # Not collected yet

    def test_metrics_property_lazy_collection(self) -> None:
        """Test that metrics property lazily collects metrics (line 64)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Add a metric to the provider
        symbol = sp.Symbol("x_1")
        metric = SymbolicMetric(
            name="count(orders)",
            symbol=symbol,
            fn=lambda k: Success(100.0),
            metric_spec=None,  # type: ignore
            dataset="orders",
            data_av_ratio=0.95,
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[symbol] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Metrics not collected yet
        assert evaluator._metrics is None

        # Access metrics property - should trigger collection
        metrics = evaluator.metrics

        # Now metrics should be collected
        assert evaluator._metrics is not None
        assert len(metrics) == 1
        assert symbol in metrics
        assert metrics[symbol] == Success(100.0)

        # Second access should return cached metrics
        metrics2 = evaluator.metrics
        assert metrics2 is metrics

    def test_collect_metrics(self) -> None:
        """Test collect_metrics method directly (lines 81-85)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Add multiple metrics
        symbols = []
        for i in range(3):
            symbol = sp.Symbol(f"x_{i + 1}")
            metric = SymbolicMetric(
                name=f"metric_{i + 1}",
                symbol=symbol,
                fn=lambda k, val=float(i + 1) * 10: Success(val),  # type: ignore
                metric_spec=None,  # type: ignore
                dataset=f"dataset_{i + 1}",
            )
            provider.registry._metrics.append(metric)
            provider.registry.index[symbol] = metric
            symbols.append(symbol)

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Collect metrics
        collected = evaluator.collect_metrics(key)

        # Verify all metrics collected
        assert len(collected) == 3
        assert collected[symbols[0]] == Success(10.0)
        assert collected[symbols[1]] == Success(20.0)
        assert collected[symbols[2]] == Success(30.0)

    def test_metric_for_symbol(self) -> None:
        """Test metric_for_symbol method."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)
        symbol = sp.Symbol("x_1")
        metric = SymbolicMetric(
            name="sum(revenue)",
            symbol=symbol,
            fn=lambda k: Success(1000.0),
            metric_spec=None,  # type: ignore
            dataset="orders",
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[symbol] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Get metric for symbol
        retrieved = evaluator.metric_for_symbol(symbol)
        assert retrieved is metric

        # Test with non-existent symbol
        unknown = sp.Symbol("unknown")
        with pytest.raises(DQXError):
            evaluator.metric_for_symbol(unknown)


class TestEvaluatorGather:
    """Test the _gather method."""

    def test_gather_with_boolean_expression(self) -> None:
        """Test _gather with non-sympy expression (line 127)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Add a metric
        symbol = sp.Symbol("x_1")
        metric = SymbolicMetric(
            name="count(users)",
            symbol=symbol,
            fn=lambda k: Success(42.0),
            metric_spec=None,  # type: ignore
            dataset="users",
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[symbol] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {symbol: Success(42.0)}

        # Pass a boolean that needs sympify
        values, infos = evaluator._gather(True)

        # Should handle boolean as constant
        assert len(values) == 0  # No symbols in True
        assert len(infos) == 0

    def test_gather_symbol_not_found(self) -> None:
        """Test _gather when symbol not in collected metrics (line 132)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Add metric to provider but not to collected metrics
        symbol = sp.Symbol("x_1")
        metric = SymbolicMetric(
            name="average(price)",
            symbol=symbol,
            fn=lambda k: Success(25.0),
            metric_spec=None,  # type: ignore
            dataset="products",
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[symbol] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Set empty metrics to trigger error
        evaluator._metrics = {}

        # Should raise DQXError
        with pytest.raises(DQXError, match=r"average.*not found in collected metrics"):
            evaluator._gather(symbol)

    def test_gather_with_mixed_success_failure(self) -> None:
        """Test _gather with both successful and failed metrics."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Add metrics
        x1 = sp.Symbol("x_1")
        x2 = sp.Symbol("x_2")

        metric1 = SymbolicMetric(
            name="sum(revenue)",
            symbol=x1,
            fn=lambda k: Success(100.0),
            metric_spec=None,  # type: ignore
            dataset="orders",
        )
        metric2 = SymbolicMetric(
            name="count(errors)",
            symbol=x2,
            fn=lambda k: Failure("Database timeout"),
            metric_spec=None,  # type: ignore
            dataset="logs",
        )

        provider.registry._metrics.append(metric1)
        provider.registry.index[x1] = metric1
        provider.registry._metrics.append(metric2)
        provider.registry.index[x2] = metric2

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={"region": "US"})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {
            x1: Success(100.0),
            x2: Failure("Database timeout"),
        }

        # Gather for expression with both symbols
        expr = x1 + x2
        values, infos = evaluator._gather(expr)

        # Values should only have successful symbol
        assert len(values) == 1
        assert values[x1] == 100.0
        assert x2 not in values

        # But infos should have both
        assert len(infos) == 2
        info_dict = {info.name: info for info in infos}

        assert "x_1" in info_dict
        assert info_dict["x_1"].metric == "sum(revenue)"
        assert info_dict["x_1"].dataset == "orders"
        assert info_dict["x_1"].value == Success(100.0)
        assert info_dict["x_1"].tags == {"region": "US"}

        assert "x_2" in info_dict
        assert info_dict["x_2"].metric == "count(errors)"
        assert info_dict["x_2"].dataset == "logs"
        assert info_dict["x_2"].value == Failure("Database timeout")


class TestEvaluatorEvaluate:
    """Test the evaluate method."""

    def test_evaluate_complex_infinity(self) -> None:
        """Test evaluation resulting in complex infinity (lines 218-219)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # No symbols needed for zoo
        evaluator._metrics = {}

        # Evaluate complex infinity (zoo)
        result = evaluator.evaluate(sp.zoo)

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert failures[0].error_message == "Validating value is infinity"
        assert failures[0].expression == "zoo"
        assert len(failures[0].symbols) == 0

    def test_evaluate_complex_number(self) -> None:
        """Test evaluation of complex numbers (lines 251-253)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Create a symbol
        x = sp.Symbol("x")
        metric = SymbolicMetric(
            name="metric",
            symbol=x,
            fn=lambda k: Success(-1.0),
            metric_spec=None,  # type: ignore
            dataset="data",
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[x] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {x: Success(-1.0)}

        # Evaluate sqrt(-1) which gives complex number
        result = evaluator.evaluate(sp.sqrt(x))

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert "complex" in failures[0].error_message
        assert "1.0i" in failures[0].error_message or "1i" in failures[0].error_message

    def test_evaluate_exception_handling(self) -> None:
        """Test exception handling during evaluation (line 241)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Create symbol with special value that causes issues
        x = sp.Symbol("x")
        metric = SymbolicMetric(
            name="metric",
            symbol=x,
            fn=lambda k: Success(1e400),  # Very large number
            metric_spec=None,  # type: ignore
            dataset="data",
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[x] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Create a custom symbol that raises during substitution
        class BadSymbol(sp.Symbol):
            def subs(self, *args: object, **kwargs: object) -> object:
                raise ValueError("Test error")

        bad_sym = BadSymbol("bad")
        evaluator._metrics = {bad_sym: Success(1.0)}

        # Mock provider to return a metric for bad symbol
        bad_metric = SymbolicMetric(
            name="bad_metric",
            symbol=bad_sym,
            fn=lambda k: Success(1.0),
            metric_spec=None,  # type: ignore
            dataset="data",
        )
        provider.registry._metrics.append(bad_metric)
        provider.registry.index[bad_sym] = bad_metric

        result = evaluator.evaluate(bad_sym)

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert "Error evaluating expression" in failures[0].error_message
        assert "Test error" in failures[0].error_message

    def test_evaluate_nan_result(self) -> None:
        """Test NaN results."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Create a symbol that will return NaN
        x = sp.Symbol("x")
        metric = SymbolicMetric(
            name="x_val",
            symbol=x,
            fn=lambda k: Success(float("nan")),  # Directly return NaN
            metric_spec=None,  # type: ignore
            dataset="data",
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[x] = metric

        evaluator._metrics = {x: Success(float("nan"))}

        # Evaluate x which is NaN
        result = evaluator.evaluate(x)

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert failures[0].error_message == "Validating value is NaN"

    def test_evaluate_infinity_result(self) -> None:
        """Test infinity results."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Create symbols
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        metrics = {
            x: SymbolicMetric(
                name="numerator",
                symbol=x,
                fn=lambda k: Success(1.0),
                metric_spec=None,  # type: ignore
                dataset="data",
            ),
            y: SymbolicMetric(
                name="denominator",
                symbol=y,
                fn=lambda k: Success(0.0),
                metric_spec=None,  # type: ignore
                dataset="data",
            ),
        }

        for metric in metrics.values():
            provider.registry._metrics.append(metric)
            provider.registry.index[metric.symbol] = metric

        evaluator._metrics = {x: Success(1.0), y: Success(0.0)}

        # 1/0 = infinity
        result = evaluator.evaluate(x / y)

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert failures[0].error_message == "Validating value is infinity"

    def test_evaluate_success(self) -> None:
        """Test successful evaluation."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Create symbols
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        for sym, name, val in [(x, "price", 10.0), (y, "quantity", 5.0)]:
            metric = SymbolicMetric(
                name=name,
                symbol=sym,
                fn=lambda k, v=val: Success(v),  # type: ignore
                metric_spec=None,  # type: ignore
                dataset="orders",
            )
            provider.registry._metrics.append(metric)
            provider.registry.index[sym] = metric

        evaluator._metrics = {x: Success(10.0), y: Success(5.0)}

        # x * y = 50.0
        result = evaluator.evaluate(x * y)

        assert isinstance(result, Success)
        assert result.unwrap() == 50.0


class TestEvaluatorDataAvailability:
    """Test data availability checking."""

    def test_check_data_availability_all_above_threshold(self) -> None:
        """Test _check_data_availability when all metrics meet threshold (lines 273-286)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Create metrics with high availability
        symbols = []
        for i in range(3):
            sym = sp.Symbol(f"x_{i + 1}")
            metric = SymbolicMetric(
                name=f"metric_{i + 1}",
                symbol=sym,
                fn=lambda k: Success(10.0),
                metric_spec=None,  # type: ignore
                dataset="data",
                data_av_ratio=0.95,  # Above threshold
            )
            provider.registry._metrics.append(metric)
            provider.registry.index[sym] = metric
            symbols.append(sym)

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Set metrics
        evaluator._metrics = {sym: Success(10.0) for sym in symbols}

        # Check availability for expression with all symbols
        expr = symbols[0] + symbols[1] * symbols[2]
        assert evaluator._check_data_availability(expr) is True

    def test_check_data_availability_below_threshold(self) -> None:
        """Test _check_data_availability when some metrics below threshold."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Create metrics with mixed availability
        x1 = sp.Symbol("x_1")
        x2 = sp.Symbol("x_2")

        metric1 = SymbolicMetric(
            name="metric_1",
            symbol=x1,
            fn=lambda k: Success(10.0),
            metric_spec=None,  # type: ignore
            dataset="data",
            data_av_ratio=0.95,  # Above threshold
        )
        metric2 = SymbolicMetric(
            name="metric_2",
            symbol=x2,
            fn=lambda k: Success(20.0),
            metric_spec=None,  # type: ignore
            dataset="data",
            data_av_ratio=0.5,  # Below threshold
        )

        provider.registry._metrics.append(metric1)
        provider.registry.index[x1] = metric1
        provider.registry._metrics.append(metric2)
        provider.registry.index[x2] = metric2

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {x1: Success(10.0), x2: Success(20.0)}

        # Check availability - should fail due to x2
        expr = x1 + x2
        assert evaluator._check_data_availability(expr) is False

    def test_check_data_availability_non_sympy_expr(self) -> None:
        """Test _check_data_availability with non-sympy expression."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {}

        # Boolean should be converted to sympy
        assert evaluator._check_data_availability(True) is True
        assert evaluator._check_data_availability(42) is True


class TestEvaluatorVisit:
    """Test the visit method for AssertionNode."""

    def test_visit_assertion_node_insufficient_data(self) -> None:
        """Test visit with insufficient data availability (lines 298-310)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Create metric with low availability
        x = sp.Symbol("x")
        metric = SymbolicMetric(
            name="count(orders)",
            symbol=x,
            fn=lambda k: Success(100.0),
            metric_spec=None,  # type: ignore
            dataset="orders",
            data_av_ratio=0.3,  # Below threshold
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[x] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {x: Success(100.0)}

        # Create assertion node
        root = RootNode("test")
        check = CheckNode(root, "check")
        validator = SymbolicValidator("> 50", lambda v: v > 50)
        assertion = AssertionNode(check, actual=x, name="Order count check", validator=validator, severity="P0")

        # Visit the assertion
        evaluator.visit(assertion)

        # Should be skipped due to low data availability
        assert assertion._result == "SKIPPED"
        assert isinstance(assertion._metric, Failure)
        failures = assertion._metric.failure()
        assert len(failures) == 1
        assert failures[0].error_message == "Insufficient data availability"

    def test_visit_assertion_node_success(self) -> None:
        """Test visit with successful evaluation and validation (lines 311-323)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Create metric with good availability
        x = sp.Symbol("x")
        metric = SymbolicMetric(
            name="revenue",
            symbol=x,
            fn=lambda k: Success(1000.0),
            metric_spec=None,  # type: ignore
            dataset="orders",
            data_av_ratio=0.95,  # Above threshold
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[x] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {x: Success(1000.0)}

        # Create assertion nodes - one that passes, one that fails
        root = RootNode("test")
        check = CheckNode(root, "check")

        # Passing assertion
        pass_validator = SymbolicValidator("> 500", lambda v: v > 500)
        pass_assertion = AssertionNode(
            check, actual=x, name="Revenue high enough", validator=pass_validator, severity="P0"
        )

        # Failing assertion
        fail_validator = SymbolicValidator("> 2000", lambda v: v > 2000)
        fail_assertion = AssertionNode(
            check, actual=x, name="Revenue over 2000", validator=fail_validator, severity="P1"
        )

        # Visit both assertions
        evaluator.visit(pass_assertion)
        evaluator.visit(fail_assertion)

        # Check passing assertion
        assert pass_assertion._result == "PASSED"
        assert isinstance(pass_assertion._metric, Success)
        assert pass_assertion._metric.unwrap() == 1000.0

        # Check failing assertion
        assert fail_assertion._result == "FAILED"
        assert isinstance(fail_assertion._metric, Success)
        assert fail_assertion._metric.unwrap() == 1000.0

    def test_visit_assertion_node_metric_failure(self) -> None:
        """Test visit when metric evaluation fails (lines 324-328)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Create metric that fails
        x = sp.Symbol("x")
        metric = SymbolicMetric(
            name="problematic_metric",
            symbol=x,
            fn=lambda k: Failure("Database connection lost"),
            metric_spec=None,  # type: ignore
            dataset="orders",
            data_av_ratio=0.95,
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[x] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {x: Failure("Database connection lost")}

        # Create assertion
        root = RootNode("test")
        check = CheckNode(root, "check")
        validator = SymbolicValidator("> 0", lambda v: v > 0)
        assertion = AssertionNode(check, actual=x, name="Metric check", validator=validator, severity="P0")

        # Visit assertion
        evaluator.visit(assertion)

        # Should fail due to metric failure
        assert assertion._result == "FAILED"
        assert isinstance(assertion._metric, Failure)

    def test_visit_assertion_node_validator_exception(self) -> None:
        """Test visit when validator raises exception."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        x = sp.Symbol("x")
        metric = SymbolicMetric(
            name="metric",
            symbol=x,
            fn=lambda k: Success(100.0),
            metric_spec=None,  # type: ignore
            dataset="data",
            data_av_ratio=0.95,
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[x] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {x: Success(100.0)}

        # Create assertion with validator that raises
        root = RootNode("test")
        check = CheckNode(root, "check")

        def bad_validator(v: float) -> bool:
            raise ValueError("Validator error")

        validator = SymbolicValidator("bad", bad_validator)
        assertion = AssertionNode(check, actual=x, name="Bad validator", validator=validator, severity="P0")

        # Visit should raise DQXError
        with pytest.raises(DQXError, match="Validator execution failed"):
            evaluator.visit(assertion)

    def test_visit_non_assertion_node(self) -> None:
        """Test visit with non-AssertionNode."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Visit non-assertion nodes - should do nothing
        root = RootNode("test")
        check = CheckNode(root, "check")

        # These should complete without error or effect
        evaluator.visit(root)
        evaluator.visit(check)


class TestEvaluatorAsync:
    """Test async functionality."""

    @pytest.mark.asyncio
    async def test_visit_async(self) -> None:
        """Test visit_async method (line 341)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Create metric
        x = sp.Symbol("x")
        metric = SymbolicMetric(
            name="async_metric",
            symbol=x,
            fn=lambda k: Success(42.0),
            metric_spec=None,  # type: ignore
            dataset="data",
            data_av_ratio=0.95,
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[x] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {x: Success(42.0)}

        # Create assertion
        root = RootNode("test")
        check = CheckNode(root, "check")
        validator = SymbolicValidator("> 0", lambda v: v > 0)
        assertion = AssertionNode(check, actual=x, name="Async test", validator=validator, severity="P0")

        # Visit async should work same as sync
        await evaluator.visit_async(assertion)

        # Should pass
        assert assertion._result == "PASSED"
        assert isinstance(assertion._metric, Success)
        assert assertion._metric.unwrap() == 42.0


class TestEvaluatorCollectSymbols:
    """Test collect_symbols method."""

    def test_collect_symbols(self) -> None:
        """Test collect_symbols method (lines 363-364)."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        # Create multiple metrics
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")

        metrics = [
            SymbolicMetric(
                name="revenue",
                symbol=x,
                fn=lambda k: Success(1000.0),
                metric_spec=None,  # type: ignore
                dataset="sales",
                data_av_ratio=0.95,
            ),
            SymbolicMetric(
                name="cost",
                symbol=y,
                fn=lambda k: Success(600.0),
                metric_spec=None,  # type: ignore
                dataset="expenses",
                data_av_ratio=0.9,
            ),
            SymbolicMetric(
                name="tax_rate",
                symbol=z,
                fn=lambda k: Failure("Tax service unavailable"),
                metric_spec=None,  # type: ignore
                dataset="tax",
                data_av_ratio=0.0,
            ),
        ]

        for metric in metrics:
            provider.registry._metrics.append(metric)
            provider.registry.index[metric.symbol] = metric

        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={"country": "US"})
        evaluator = Evaluator(provider, key, "Financial Suite", data_av_threshold=0.8)

        # Set up metrics
        evaluator._metrics = {
            x: Success(1000.0),
            y: Success(600.0),
            z: Failure("Tax service unavailable"),
        }

        # Create expression with all symbols
        expr = (x - y) * z

        # Collect symbols
        symbol_infos = evaluator.collect_symbols(expr)

        # Should have all 3 symbols
        assert len(symbol_infos) == 3

        # Create lookup by name
        info_dict = {info.name: info for info in symbol_infos}

        # Check x
        assert "x" in info_dict
        assert info_dict["x"].metric == "revenue"
        assert info_dict["x"].dataset == "sales"
        assert info_dict["x"].value == Success(1000.0)
        assert info_dict["x"].yyyy_mm_dd == dt.date(2024, 1, 1)
        assert info_dict["x"].tags == {"country": "US"}

        # Check y
        assert "y" in info_dict
        assert info_dict["y"].metric == "cost"
        assert info_dict["y"].dataset == "expenses"
        assert info_dict["y"].value == Success(600.0)

        # Check z - failed metric
        assert "z" in info_dict
        assert info_dict["z"].metric == "tax_rate"
        assert info_dict["z"].dataset == "tax"
        assert info_dict["z"].value == Failure("Tax service unavailable")

    def test_collect_symbols_empty_expression(self) -> None:
        """Test collect_symbols with constant expression."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)
        evaluator._metrics = {}

        # Constant expression has no symbols
        expr = sp.sympify(100)

        symbol_infos = evaluator.collect_symbols(expr)

        # Should be empty
        assert len(symbol_infos) == 0
