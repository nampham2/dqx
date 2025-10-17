"""Additional tests to improve coverage for api.py."""

import datetime

import pytest
import sympy as sp

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import DQXError, ResultKey, SymbolicValidator
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
    from dqx.api import AssertionReady

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
        suite.run({}, key)  # Empty datasources dict


def test_collect_results_before_run_error() -> None:
    """Test collect_results raises error when called before run()."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    with pytest.raises(DQXError, match="Cannot collect results before suite execution"):
        suite.collect_results()


def test_collect_symbols_before_run_error() -> None:
    """Test collect_symbols raises error when called before run()."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    with pytest.raises(DQXError, match="Cannot collect symbols before suite execution"):
        suite.collect_symbols()


def test_verification_suite_already_executed_error() -> None:
    """Test VerificationSuite.run raises error when suite already executed."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # We just need to set is_evaluated to True to test the error
    suite.is_evaluated = True

    # Second run should raise error
    with pytest.raises(DQXError, match="Verification suite has already been executed"):
        # Use a mock datasource that doesn't need actual implementation
        suite.run({"test_ds": None}, key)  # type: ignore


def test_verification_suite_multiple_checks() -> None:
    """Test VerificationSuite with multiple checks to ensure proper graph building."""
    db = InMemoryMetricDB()

    @check(name="Check 1")
    def check1(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    @check(name="Check 2", tags=["important"])
    def check2(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("price")).where(name="Price is positive").is_positive()

    @check(name="Check 3", datasets=["orders"])
    def check3(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.sum("quantity")).where(name="Total quantity").is_geq(100)

    suite = VerificationSuite([check1, check2, check3], db, "Multi Check Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Build graph
    suite.build_graph(suite._context, key)

    # Verify all checks were added
    checks = list(suite.graph.checks())
    assert len(checks) == 3
    assert {c.name for c in checks} == {"Check 1", "Check 2", "Check 3"}

    # Verify tags and datasets
    check2_node = next(c for c in checks if c.name == "Check 2")
    assert check2_node.tags == ["important"]

    check3_node = next(c for c in checks if c.name == "Check 3")
    assert check3_node.datasets == ["orders"]


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
    suite.is_evaluated = True
    suite._key = None

    with pytest.raises(DQXError, match="No ResultKey available"):
        suite.collect_results()


def test_verification_suite_collect_symbols_without_key() -> None:
    """Test collect_symbols with None key (edge case that should not happen)."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="Test").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Force the state where suite is evaluated but key is None
    # This is an edge case that shouldn't happen in normal operation
    suite.is_evaluated = True
    suite._key = None

    with pytest.raises(DQXError, match="No ResultKey available"):
        suite.collect_symbols()
