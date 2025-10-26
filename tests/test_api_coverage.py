"""Additional tests to improve coverage for api.py."""

import datetime
from typing import Any

import pytest
import sympy as sp

from dqx.api import AssertionReady, Context, MetricProvider, VerificationSuite, check
from dqx.common import DQXError, ResultKey, ResultKeyProvider, SymbolicValidator
from dqx.orm.repositories import InMemoryMetricDB


def test_is_negative_assertion() -> None:
    """Test is_negative assertion method."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Negative Check")
    def negative_check(mp: MetricProvider, ctx: Context) -> None:
        # Test is_negative
        ctx.assert_that(sp.Symbol("x")).where(name="X is negative").is_negative()

        # Verify assertion was created
        assert ctx.current_check is not None
        assert len(ctx.current_check.children) == 1
        assert ctx.current_check.children[0].name == "X is negative"

    suite = VerificationSuite([negative_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key=key)


def test_create_assertion_node_with_none_context() -> None:
    """Test _create_assertion_node when context is None."""

    # Create AssertionReady with None context
    ready = AssertionReady(
        actual=sp.Symbol("x"),
        name="Test assertion",
        severity="P1",
        context=None,  # This is the key - None context
    )

    # Should not raise error, just return early
    validator = SymbolicValidator("> 0", lambda x: x > 0)
    ready._create_assertion_node(validator)  # Should return without error


def test_create_assertion_node_outside_check_context() -> None:
    """Test _create_assertion_node raises error when not in check context."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    # Create AssertionReady with a context but no active check
    ready = context.assert_that(sp.Symbol("x")).where(name="Test assertion")

    # Should raise error since we're not inside a check
    with pytest.raises(DQXError, match="Cannot create assertion outside of check context"):
        ready.is_positive()


def test_verification_suite_empty_checks_error() -> None:
    """Test VerificationSuite raises error with empty checks list."""
    db = InMemoryMetricDB()

    with pytest.raises(DQXError, match="At least one check must be provided"):
        VerificationSuite([], db, "Test Suite")


def test_verification_suite_empty_name_error() -> None:
    """Test VerificationSuite raises error with empty name."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        pass

    with pytest.raises(DQXError, match="Suite name cannot be empty"):
        VerificationSuite([test_check], db, "")

    with pytest.raises(DQXError, match="Suite name cannot be empty"):
        VerificationSuite([test_check], db, "   ")


def test_verification_suite_run_no_datasources_error() -> None:
    """Test VerificationSuite.run raises error with no datasources."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    with pytest.raises(DQXError, match="No data sources provided"):
        suite.run([], key)  # Empty datasources list


def test_collect_results_before_run_error() -> None:
    """Test collect_results raises error when called before run()."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    with pytest.raises(DQXError, match="Verification suite has not been executed yet!"):
        suite.collect_results()


def test_collect_symbols_before_run_error() -> None:
    """Test that accessing symbols before run() is not possible."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Since collect_symbols is on provider, we need to ensure we can't get meaningful
    # symbols before run. The provider exists but won't have evaluated metrics.
    # This test verifies the concept that symbols aren't available before run.
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    symbols = suite.provider.collect_symbols(key)

    # Should return empty list as no metrics have been evaluated
    assert symbols == []


def test_verification_suite_already_executed_error() -> None:
    """Test VerificationSuite.run raises error when suite already executed."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # We just need to set is_evaluated to True to test the error
    suite._is_evaluated = True

    # Second run should raise error
    with pytest.raises(DQXError, match="Verification suite has already been executed"):
        # Use a mock datasource that doesn't need actual implementation
        import pyarrow as pa

        from dqx.datasource import DuckRelationDataSource

        data = pa.table({"price": [10, 20, 30]})
        suite.run([DuckRelationDataSource.from_arrow(data, "data")], key)


def test_verification_suite_multiple_checks() -> None:
    """Test VerificationSuite with multiple checks to ensure proper graph building."""
    db = InMemoryMetricDB()

    @check(name="Check 1")
    def check1(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    @check(name="Check 2")
    def check2(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("price")).where(name="Price is positive").is_positive()

    @check(name="Check 3", datasets=["data"])
    def check3(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.sum("quantity")).where(name="Total quantity").is_geq(100)

    suite = VerificationSuite([check1, check2, check3], db, "Multi Check Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run suite which will build graph internally
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    data = pa.table({"price": [10, 20, 30], "quantity": [100, 200, 300]})
    suite.run([DuckRelationDataSource.from_arrow(data, "data")], key)

    # Verify all checks were added
    checks = list(suite.graph.checks())
    assert len(checks) == 3
    assert {c.name for c in checks} == {"Check 1", "Check 2", "Check 3"}

    # Verify datasets (tags have been removed)
    check3_node = next(c for c in checks if c.name == "Check 3")
    assert check3_node.datasets == ["data"]


def test_context_provider_property() -> None:
    """Test Context.provider property returns MetricProvider."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    assert isinstance(context.provider, MetricProvider)
    assert context.provider is context._provider


def test_verification_suite_provider_property() -> None:
    """Test VerificationSuite.provider property returns MetricProvider."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        pass

    suite = VerificationSuite([test_check], db, "Test Suite")

    assert isinstance(suite.provider, MetricProvider)
    assert suite.provider is suite._context.provider


def test_verification_suite_collect_results_without_key() -> None:
    """Test collect_results with None key (edge case that should not happen)."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Force the state where suite is evaluated but key is None
    # This is an edge case that shouldn't happen in normal operation
    suite._is_evaluated = True
    suite._key = None

    with pytest.raises(DQXError, match="No ResultKey available"):
        suite.collect_results()


def test_verification_suite_collect_symbols_without_key() -> None:
    """Test that symbols can't be collected properly without a key."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        # Use a metric that will register a symbolic metric
        ctx.assert_that(mp.average("price")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Build the graph to register the symbolic metric
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(suite._context, key)

    # Force the state where suite is evaluated but key is None
    # This is an edge case that shouldn't happen in normal operation
    suite._is_evaluated = True
    suite._key = None

    # Try to collect symbols - the provider method needs a key
    with pytest.raises(DQXError, match="No ResultKey available"):
        # Access the key property which will raise the error
        _ = suite.key


def test_verification_suite_graph_before_build_error() -> None:
    """Test accessing graph property before build_graph() raises error."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Accessing graph before build_graph() should raise error
    with pytest.raises(DQXError, match="Verification suite has not been executed yet!"):
        _ = suite.graph


def test_verification_suite_plugin_manager_lazy_initialization() -> None:
    """Test plugin_manager property lazy initialization."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Initially, _plugin_manager should be None
    assert suite._plugin_manager is None

    # Access plugin_manager property - should create instance
    plugin_manager = suite.plugin_manager

    # Verify it's created
    assert plugin_manager is not None
    assert suite._plugin_manager is not None

    # Verify it returns the same instance on subsequent access
    assert suite.plugin_manager is plugin_manager


def test_collect_symbols_with_evaluation_error() -> None:
    """Test collect_symbols handles exceptions during symbol evaluation."""
    from unittest.mock import patch

    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        # This will register a symbolic metric
        ctx.assert_that(mp.average("price")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Build the graph to register metrics
    suite.build_graph(suite._context, key)

    # Mark as evaluated
    suite._is_evaluated = True
    suite._key = key

    # Get the registered metric
    assert len(suite._context.provider.symbolic_metrics) > 0
    metric = suite._context.provider.symbolic_metrics[0]

    # Mock the fn to raise an exception
    with patch.object(metric, "fn", side_effect=RuntimeError("Test error")):
        # Collect symbols via provider
        symbols = suite.provider.collect_symbols(key)

        # Verify the symbol was created with a Failure value
        assert len(symbols) == 1
        assert symbols[0].name == "x_1"
        # Check if it's a Failure instance
        from returns.result import Failure

        assert isinstance(symbols[0].value, Failure)


def test_assertion_name_validation() -> None:
    """Test assertion name validation in where() method."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    # Test empty name
    with pytest.raises(ValueError, match="Assertion name cannot be empty"):
        context.assert_that(sp.Symbol("x")).where(name="")

    # Test whitespace-only name
    with pytest.raises(ValueError, match="Assertion name cannot be empty"):
        context.assert_that(sp.Symbol("x")).where(name="   ")

    # Test name that's too long (>255 chars)
    long_name = "a" * 256
    with pytest.raises(ValueError, match="Assertion name is too long"):
        context.assert_that(sp.Symbol("x")).where(name=long_name)


def test_is_leq_assertion() -> None:
    """Test is_leq (less than or equal) assertion method."""
    db = InMemoryMetricDB()

    @check(name="LEQ Check")
    def leq_check(mp: MetricProvider, ctx: Context) -> None:
        # Test is_leq
        ctx.assert_that(mp.average("x")).where(name="X is LEQ 10").is_leq(10)
        ctx.assert_that(mp.average("y")).where(name="Y is LEQ 20 with tol").is_leq(20, tol=0.1)

    suite = VerificationSuite([leq_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run suite which will build graph internally
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    data = pa.table({"x": [1, 2, 3], "y": [10, 15, 20]})
    suite.run([DuckRelationDataSource.from_arrow(data, "data")], key)

    # Verify assertions were created
    assertions = list(suite.graph.assertions())
    assert len(assertions) == 2
    assert assertions[0].name == "X is LEQ 10"
    assert assertions[1].name == "Y is LEQ 20 with tol"


def test_is_lt_assertion() -> None:
    """Test is_lt (less than) assertion method."""
    db = InMemoryMetricDB()

    @check(name="LT Check")
    def lt_check(mp: MetricProvider, ctx: Context) -> None:
        # Test is_lt
        ctx.assert_that(mp.average("x")).where(name="X is LT 10").is_lt(10)
        ctx.assert_that(mp.average("y")).where(name="Y is LT 20 with tol").is_lt(20, tol=0.01)

    suite = VerificationSuite([lt_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run suite which will build graph internally
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    data = pa.table({"x": [5, 10, 15], "y": [18, 19, 20]})
    suite.run([DuckRelationDataSource.from_arrow(data, "data")], key)

    # Verify assertions were created
    assertions = list(suite.graph.assertions())
    assert len(assertions) == 2
    assert assertions[0].name == "X is LT 10"
    assert assertions[1].name == "Y is LT 20 with tol"


def test_is_eq_assertion() -> None:
    """Test is_eq (equal) assertion method."""
    db = InMemoryMetricDB()

    @check(name="EQ Check")
    def eq_check(mp: MetricProvider, ctx: Context) -> None:
        # Test is_eq
        ctx.assert_that(mp.average("x")).where(name="X equals 10").is_eq(10)
        ctx.assert_that(mp.average("y")).where(name="Y equals 20 with tol").is_eq(20, tol=0.001)

    suite = VerificationSuite([eq_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run suite which will build graph internally
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    data = pa.table({"x": [10, 10, 10], "y": [20, 20, 20]})
    suite.run([DuckRelationDataSource.from_arrow(data, "data")], key)

    # Verify assertions were created
    assertions = list(suite.graph.assertions())
    assert len(assertions) == 2
    assert assertions[0].name == "X equals 10"
    assert assertions[1].name == "Y equals 20 with tol"


def test_is_between_invalid_range() -> None:
    """Test is_between with invalid range (lower > upper)."""
    db = InMemoryMetricDB()

    # Test invalid range where lower > upper
    # The error should be raised when is_between is called
    with pytest.raises(
        ValueError, match="Invalid range: lower bound \\(10\\) must be less than or equal to upper bound \\(5\\)"
    ):
        # Create the assertion ready object first
        context = Context("test", db)
        ready = context.assert_that(sp.Symbol("x")).where(name="Invalid range")
        # This should raise the ValueError
        ready.is_between(10, 5)


def test_is_between_valid_assertion() -> None:
    """Test is_between with valid range."""
    db = InMemoryMetricDB()

    @check(name="Between Check")
    def between_check(mp: MetricProvider, ctx: Context) -> None:
        # Test valid range
        ctx.assert_that(mp.average("x")).where(name="X is between 0 and 10").is_between(0, 10)
        ctx.assert_that(mp.average("y")).where(name="Y is between -5 and 5 with tol").is_between(-5, 5, tol=0.1)

    suite = VerificationSuite([between_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run suite which will build graph internally
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    data = pa.table({"x": [5, 7, 9], "y": [-3, 0, 3]})
    suite.run([DuckRelationDataSource.from_arrow(data, "data")], key)

    # Verify assertions were created
    assertions = list(suite.graph.assertions())
    assert len(assertions) == 2
    assert assertions[0].name == "X is between 0 and 10"
    assert assertions[1].name == "Y is between -5 and 5 with tol"


def test_context_pending_metrics_with_dataset() -> None:
    """Test Context.pending_metrics with dataset filter."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    # Register some metrics with different datasets
    mp = context.provider
    mp.sum("col1", dataset="ds1")
    mp.average("col2", dataset="ds2")
    mp.num_rows(dataset="ds1")
    mp.sum("col4")  # No dataset - using sum instead of max

    # Get metrics for specific dataset
    ds1_metrics = context.pending_metrics("ds1")
    assert len(ds1_metrics) == 2
    assert all(m.dataset == "ds1" for m in ds1_metrics)

    ds2_metrics = context.pending_metrics("ds2")
    assert len(ds2_metrics) == 1
    assert ds2_metrics[0].dataset == "ds2"

    # Get all metrics
    all_metrics = context.pending_metrics()
    assert len(all_metrics) == 4


def test_context_key_property() -> None:
    """Test Context.key property returns ResultKeyProvider."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    # Access key property
    key_provider = context.key

    # Should return a ResultKeyProvider instance
    assert isinstance(key_provider, ResultKeyProvider)


def test_suite_validation_with_errors() -> None:
    """Test suite validation that produces errors."""
    from unittest.mock import MagicMock, patch

    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Mock the validator instance and its validate method
    mock_validator_instance = MagicMock()
    mock_report = MagicMock()
    setattr(mock_report, "has_errors", MagicMock(return_value=True))
    setattr(mock_report, "__str__", MagicMock(return_value="Validation error: Missing dataset"))
    mock_validator_instance.validate.return_value = mock_report

    # Mock the SuiteValidator class
    with patch("dqx.api.SuiteValidator", return_value=mock_validator_instance):
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Should raise DQXError with the validation report
        with pytest.raises(DQXError, match=r"Suite validation failed:\nValidation error: Missing dataset"):
            suite.build_graph(suite._context, key)


def test_suite_validation_with_warnings() -> None:
    """Test suite validation that produces warnings but no errors."""
    import logging
    from unittest.mock import MagicMock, patch

    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Mock the validator instance and its validate method
    mock_validator_instance = MagicMock()
    mock_report = MagicMock()
    setattr(mock_report, "has_errors", MagicMock(return_value=False))
    setattr(mock_report, "has_warnings", MagicMock(return_value=True))
    setattr(mock_report, "__str__", MagicMock(return_value="Warning: Unused dataset"))
    mock_validator_instance.validate.return_value = mock_report

    # Mock the SuiteValidator class
    with patch("dqx.api.SuiteValidator", return_value=mock_validator_instance):
        # Capture log output
        with patch.object(logging.getLogger("dqx.api"), "warning") as mock_warning:
            key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
            suite.build_graph(suite._context, key)

            # Should log the warning
            mock_warning.assert_called_once()
            assert "Suite validation warnings:" in mock_warning.call_args[0][0]
            assert "Warning: Unused dataset" in mock_warning.call_args[0][0]


def test_verification_suite_run_full_execution() -> None:
    """Test full execution of VerificationSuite.run() method covering all code paths."""
    from unittest.mock import MagicMock, PropertyMock, patch

    db = InMemoryMetricDB()

    # Create checks with different datasets and key providers
    @check(name="Check 1", datasets=["ds1"])
    def check1(mp: MetricProvider, ctx: Context) -> None:
        # This will create a metric for ds1
        ctx.assert_that(mp.sum("col1")).where(name="Sum col1").is_positive()

    @check(name="Check 2", datasets=["ds2"])
    def check2(mp: MetricProvider, ctx: Context) -> None:
        # This will create a metric for ds2
        ctx.assert_that(mp.average("col2")).where(name="Avg col2").is_gt(10)

    @check(name="Check 3", datasets=["ds1"])
    def check3(mp: MetricProvider, ctx: Context) -> None:
        # This will create a metric for ds1 with lag
        ctx.assert_that(mp.num_rows(lag=1)).where(name="Count rows lag").is_geq(100)

    suite = VerificationSuite([check1, check2, check3], db, "Test Suite")

    # Create mock datasources that implement SqlDataSource protocol
    mock_ds1 = MagicMock()
    type(mock_ds1).name = PropertyMock(return_value="ds1")
    mock_ds1.select = MagicMock()

    mock_ds2 = MagicMock()
    type(mock_ds2).name = PropertyMock(return_value="ds2")
    mock_ds2.select = MagicMock()

    datasources = [mock_ds1, mock_ds2]

    # Mock the Analyzer to track calls
    mock_analyzer = MagicMock()
    mock_report = MagicMock()
    mock_analyzer.report = mock_report

    # Track analyze calls
    analyze_calls = []

    def track_analyze(ds: Any, metrics_by_key: Any) -> Any:
        # Track which datasource was analyzed and what keys were used
        for key, metrics in metrics_by_key.items():
            analyze_calls.append((ds.name, len(metrics), key.yyyy_mm_dd))
        return MagicMock()  # Return a mock report

    mock_analyzer.analyze.side_effect = track_analyze

    # Mock the Evaluator
    mock_evaluator = MagicMock()

    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={"env": "test"})

    with patch("dqx.api.Analyzer", return_value=mock_analyzer):
        with patch("dqx.api.Evaluator", return_value=mock_evaluator):
            # Run the suite
            suite.run(datasources, key, enable_plugins=False)  # type: ignore[arg-type]

    # Verify the suite was executed
    assert suite._is_evaluated is True
    assert suite._key == key

    # Verify analyzer was called correctly
    # The analyze calls contain all the metrics grouped by key
    assert len(analyze_calls) == 3  # Total of 3 key/datasource combinations

    # Check ds1 was analyzed - with metrics for two different dates (current and lag)
    ds1_calls = [c for c in analyze_calls if c[0] == "ds1"]
    assert len(ds1_calls) == 2  # Two different dates for ds1
    # One call should be for current date, one for lagged date
    dates = {c[2] for c in ds1_calls}
    assert len(dates) == 2  # Two different dates

    # Check ds2 was analyzed with 1 metric
    ds2_calls = [c for c in analyze_calls if c[0] == "ds2"]
    assert len(ds2_calls) == 1
    assert ds2_calls[0][1] == 1  # 1 metric

    # Verify analyze was called once per datasource
    assert mock_analyzer.analyze.call_count == 2  # Once for ds1, once for ds2

    # Verify evaluator was used
    assert mock_evaluator.visit.called


def test_verification_suite_run_with_plugins() -> None:
    """Test VerificationSuite.run() with plugins enabled."""
    from unittest.mock import MagicMock, PropertyMock, patch

    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Create mock datasource that implements SqlDataSource protocol
    mock_ds = MagicMock()
    type(mock_ds).name = PropertyMock(return_value="test_ds")
    mock_ds.select = MagicMock()
    datasources = [mock_ds]

    # Mock the Analyzer and Evaluator
    mock_analyzer = MagicMock()
    mock_analyzer.report = MagicMock()
    mock_evaluator = MagicMock()

    # Track if _process_plugins was called
    process_plugins_called = False

    def mock_process_plugins(ds: Any) -> None:
        nonlocal process_plugins_called
        process_plugins_called = True
        # Call original with mocked plugin manager
        suite._plugin_manager = MagicMock()
        suite._plugin_manager.process_all = MagicMock()

    suite._process_plugins = mock_process_plugins  # type: ignore[assignment]

    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    with patch("dqx.api.Analyzer", return_value=mock_analyzer):
        with patch("dqx.api.Evaluator", return_value=mock_evaluator):
            # Run with plugins enabled
            suite.run(datasources, key, enable_plugins=True)  # type: ignore[arg-type]

    # Verify plugins were processed
    assert process_plugins_called


def test_collect_results_successful() -> None:
    """Test successful collection of results after suite execution."""
    from unittest.mock import MagicMock

    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.sum("amount")).where(name="Total amount positive", severity="P0").is_positive()
        ctx.assert_that(mp.average("price")).where(name="Average price reasonable").is_between(10, 1000)

    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={"env": "prod"})

    # Build graph
    suite.build_graph(suite._context, key)

    # Mark as evaluated
    suite._is_evaluated = True
    suite._key = key

    # Mock assertion nodes to have results
    assertions = list(suite.graph.assertions())
    assert len(assertions) == 2

    # Set mock results on assertions
    assertions[0]._result = "OK"
    assertions[0]._metric = MagicMock()
    assertions[0]._metric.is_success = True

    assertions[1]._result = "FAILURE"
    assertions[1]._metric = MagicMock()
    assertions[1]._metric.is_failure = True

    # Collect results
    results = suite.collect_results()

    # Verify results
    assert len(results) == 2

    # Check first result
    assert results[0].suite == "Test Suite"
    assert results[0].check == "Test Check"
    assert results[0].assertion == "Total amount positive"
    assert results[0].severity == "P0"
    assert results[0].status == "OK"
    assert results[0].yyyy_mm_dd == key.yyyy_mm_dd
    assert results[0].tags == {"env": "prod"}

    # Check second result
    assert results[1].assertion == "Average price reasonable"
    assert results[1].status == "FAILURE"
    assert results[1].severity == "P1"  # Default severity
