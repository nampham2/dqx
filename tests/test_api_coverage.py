"""Additional tests to improve coverage for api.py."""

import datetime
import uuid

import pytest
import sympy as sp

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.orm.repositories import InMemoryMetricDB


def test_context_pending_metrics() -> None:
    """Test Context.pending_metrics method."""
    db = InMemoryMetricDB()
    execution_id = str(uuid.uuid4())
    context = Context("test", db, execution_id)

    # Create some metrics through the provider
    provider = context.provider
    avg_metric = provider.average("revenue")
    sum_metric = provider.sum("orders")

    # Get pending metrics
    pending = context.pending_metrics()

    # Should have the metrics we created
    assert len(pending) == 2

    # Check the symbols are present in the pending metrics
    pending_symbols = [m.symbol for m in pending]
    assert avg_metric in pending_symbols
    assert sum_metric in pending_symbols


def test_context_provider_property() -> None:
    """Test Context.provider property."""
    db = InMemoryMetricDB()
    execution_id = str(uuid.uuid4())
    context = Context("test", db, execution_id)

    # Provider should be accessible
    assert context.provider is not None
    assert isinstance(context.provider, MetricProvider)
    assert context.provider._db is db
    assert context.provider._execution_id == execution_id


def test_context_key_property() -> None:
    """Test Context.key property."""
    db = InMemoryMetricDB()
    execution_id = str(uuid.uuid4())
    context = Context("test", db, execution_id)

    # Context.key returns a ResultKeyProvider
    from dqx.common import ResultKeyProvider

    assert isinstance(context.key, ResultKeyProvider)


def test_create_assertion_node_outside_check_context() -> None:
    """Test creating assertion node outside check context."""
    db = InMemoryMetricDB()
    execution_id = str(uuid.uuid4())
    context = Context("test", db, execution_id)

    # Should raise error when not in check context
    from dqx.common import DQXError

    with pytest.raises(DQXError, match="Cannot create assertion outside"):
        # Try to create assertion outside of check
        context.assert_that(sp.Symbol("x")).where(name="test").is_positive()


def test_assertion_name_validation() -> None:
    """Test assertion name validation in check context."""
    db = InMemoryMetricDB()

    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    pa_table = pa.table({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # Test that empty assertion name raises error
    @check(name="Test Check")
    def test_check_empty(mp: MetricProvider, ctx: Context) -> None:
        # This should raise an error
        ctx.assert_that(mp.average("col1")).where(name="", severity="P1").is_gt(0)

    suite = VerificationSuite([test_check_empty], db, "test")
    datasource = DuckRelationDataSource.from_arrow(pa_table, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Should raise ValueError for empty name
    with pytest.raises(ValueError, match="Assertion name cannot be empty"):
        suite.run([datasource], key)

    # Test that too long name also raises error
    @check(name="Test Check Long")
    def test_check_long(mp: MetricProvider, ctx: Context) -> None:
        # Name longer than 255 characters
        long_name = "a" * 256
        ctx.assert_that(mp.average("col1")).where(name=long_name, severity="P1").is_gt(0)

    suite2 = VerificationSuite([test_check_long], db, "test2")

    # Should raise ValueError for too long name
    with pytest.raises(ValueError, match="Assertion name is too long"):
        suite2.run([datasource], key)


def test_is_negative_assertion() -> None:
    """Test is_negative assertion method."""
    db = InMemoryMetricDB()

    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    pa_table = pa.table({"col1": [-1, -2, -3], "col2": [1, 2, 3]})

    @check(name="Negative Check")
    def negative_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("col1")).where(name="Col1 avg is negative").is_negative()
        ctx.assert_that(mp.average("col2")).where(name="Col2 avg is negative").is_negative()

    suite = VerificationSuite([negative_check], db, "test")
    datasource = DuckRelationDataSource.from_arrow(pa_table, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    suite.run([datasource], key)
    results = suite.collect_results()

    # First assertion should pass, second should fail
    assert len(results) == 2
    assert results[0].status == "OK"
    assert results[1].status == "FAILURE"


def test_is_between_invalid_range() -> None:
    """Test is_between with invalid range (lower > upper)."""
    db = InMemoryMetricDB()

    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    pa_table = pa.table({"col1": [1, 2, 3]})

    @check(name="Invalid Range")
    def invalid_check(mp: MetricProvider, ctx: Context) -> None:
        # This should raise an error
        ctx.assert_that(mp.average("col1")).where(name="test").is_between(10, 5)

    suite = VerificationSuite([invalid_check], db, "test")
    datasource = DuckRelationDataSource.from_arrow(pa_table, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Should raise error for invalid range
    with pytest.raises(ValueError, match="lower bound .* must be less than or equal to upper bound"):
        suite.run([datasource], key)


def test_context_pending_metrics_with_dataset() -> None:
    """Test Context.pending_metrics with dataset filter."""
    db = InMemoryMetricDB()
    execution_id = str(uuid.uuid4())
    context = Context("test", db, execution_id)

    # Create metrics with different datasets
    provider = context.provider
    metric1 = provider.average("revenue", dataset="sales")
    metric2 = provider.sum("orders", dataset="sales")
    metric3 = provider.minimum("price", dataset="products")

    # Get all pending metrics
    all_pending = context.pending_metrics()
    assert len(all_pending) == 3

    # Get metrics for specific dataset
    sales_pending = context.pending_metrics(dataset="sales")
    assert len(sales_pending) == 2

    # Check the symbols are present in the pending metrics
    sales_symbols = [m.symbol for m in sales_pending]
    assert metric1 in sales_symbols
    assert metric2 in sales_symbols
    assert metric3 not in sales_symbols

    products_pending = context.pending_metrics(dataset="products")
    assert len(products_pending) == 1
    products_symbols = [m.symbol for m in products_pending]
    assert metric3 in products_symbols


def test_verification_suite_edge_cases() -> None:
    """Test VerificationSuite edge cases."""
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Test with no checks - should raise DQXError
    from dqx.common import DQXError

    with pytest.raises(DQXError, match="At least one check must be provided"):
        suite = VerificationSuite([], db, "no checks")

    # Test with no data sources
    @check(name="dummy")
    def dummy_check(mp: MetricProvider, ctx: Context) -> None:
        pass

    suite = VerificationSuite([dummy_check], db, "test")
    with pytest.raises(DQXError, match="No data sources"):
        suite.run([], key)


def test_check_decorator() -> None:
    """Test check decorator functionality."""
    db = InMemoryMetricDB()

    # Test that check decorator works
    @check(name="Test Decorator")
    def test_decorator_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("x")).where(name="X is positive").is_positive()

    # Check has the right name
    assert hasattr(test_decorator_check, "__name__")

    # Can be used in VerificationSuite
    suite = VerificationSuite([test_decorator_check], db, "test")
    assert suite._name == "test"


def test_metric_provider_symbol_tracking() -> None:
    """Test that MetricProvider properly tracks created symbols."""
    db = InMemoryMetricDB()
    execution_id = str(uuid.uuid4())
    provider = MetricProvider(db, execution_id)

    # Create various metrics
    avg = provider.average("col1")
    sum_metric = provider.sum("col2")
    min_metric = provider.minimum("col3")

    # Check that metrics are tracked
    metrics = provider.metrics
    assert len(metrics) == 3

    # Check the symbols are created
    symbols = [m.symbol for m in metrics]
    assert avg in symbols
    assert sum_metric in symbols
    assert min_metric in symbols


def test_result_key_with_date() -> None:
    """Test ResultKey with date parameter."""
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={"env": "prod"})
    assert key.yyyy_mm_dd == datetime.date(2024, 1, 1)
    assert key.tags == {"env": "prod"}

    # Test string representation
    assert str(key) == "ResultKey(2024-01-01, {'env': 'prod'})"


def test_context_with_key() -> None:
    """Test Context key property."""
    db = InMemoryMetricDB()
    execution_id = str(uuid.uuid4())
    context = Context("test", db, execution_id)

    # Key property should return ResultKeyProvider
    key = context.key
    assert key is not None

    # Can create ResultKey from it
    result_key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
    assert result_key.yyyy_mm_dd == datetime.date(2024, 1, 1)
    assert result_key.tags == {}
