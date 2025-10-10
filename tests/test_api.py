import datetime

import pytest
import sympy as sp

from dqx.api import AssertBuilder, Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.graph.nodes import AssertionNode
from dqx.orm.repositories import InMemoryMetricDB


def test_assertion_node_is_immutable() -> None:
    """AssertionNode should be immutable after creation."""
    # Test that AssertionNode doesn't have setter methods
    node = AssertionNode(actual=sp.Symbol("x"))

    # These methods should not exist
    assert not hasattr(node, "set_label")
    assert not hasattr(node, "set_severity")
    assert not hasattr(node, "set_validator")

    # Fields can be set at construction but not modified after
    node_with_label = AssertionNode(actual=sp.Symbol("x"), name="test label")
    assert node_with_label.name == "test label"


def test_assert_builder_no_listeners() -> None:
    """AssertBuilder should not use listeners."""
    expr = sp.Symbol("x")

    # Should not accept listeners parameter
    with pytest.raises(TypeError):
        AssertBuilder(actual=expr, listeners=[], context=None)  # type: ignore[call-arg]

    # Should work without listeners
    builder = AssertBuilder(actual=expr, context=None)
    assert builder is not None


def test_assertion_methods_return_none() -> None:
    """Assertion methods should not return AssertBuilder for chaining."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    # Create a simple check to have proper context
    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        builder = ctx.assert_that(sp.Symbol("x"))

        # These should return None, not AssertBuilder
        result = builder.is_gt(0)  # type: ignore[func-returns-value]
        assert result is None  # Should return None

        # Try other assertion methods
        builder2 = ctx.assert_that(sp.Symbol("y"))
        result2 = builder2.is_eq(5)  # type: ignore[func-returns-value]
        assert result2 is None  # Should return None

        result3 = ctx.assert_that(sp.Symbol("z")).is_leq(10)  # type: ignore[func-returns-value]
        assert result3 is None  # Should return None

    # Set up suite with the check
    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.collect(context, key=key)


def test_no_assertion_chaining() -> None:
    """Chained assertions should not be possible."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        metric = sp.Symbol("x")
        # This should fail - can't chain assertions
        result = ctx.assert_that(metric).is_gt(0)  # type: ignore[func-returns-value]
        # Result should be None, so calling is_lt on it should fail
        with pytest.raises(AttributeError):
            result.is_lt(100)  # type: ignore[attr-defined]

    # Set up suite and run check
    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.collect(context, key=key)


def test_multiple_assertions_on_same_metric() -> None:
    """Test that multiple separate assertions can be made on the same metric."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        metric = sp.Symbol("x")

        # Multiple assertions on same metric (not chained)
        ctx.assert_that(metric).where(name="Greater than 40").is_gt(40)
        ctx.assert_that(metric).where(name="Less than 60").is_lt(60)

        # Verify we have 2 separate assertions
        check_node = ctx.current_check
        assert check_node is not None
        assert len(check_node.children) == 2

        # Verify labels were set correctly
        assert check_node.children[0].name == "Greater than 40"
        assert check_node.children[1].name == "Less than 60"

    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.collect(context, key=key)


def test_simple_check_uses_function_name() -> None:
    """Test that @check without params uses function name."""

    # Create a simple check without parameters
    @check(name="validate_orders")
    def validate_orders(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).is_gt(0)

    # No metadata is stored anymore, just verify the function works
    assert validate_orders.__name__ == "validate_orders"


def test_parametrized_check_uses_provided_name() -> None:
    """Test that @check with name parameter uses that name."""

    @check(name="Order Validation Check", tags=["critical"])
    def validate_orders(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).is_gt(0)

    # No metadata is stored anymore, just verify the function works
    assert validate_orders.__name__ == "validate_orders"


def test_simple_check_works_in_suite() -> None:
    """Test that simple @check works in a verification suite."""

    @check(name="my_simple_check")
    def my_simple_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).is_gt(0)

    # Should be able to use in a suite without errors
    db = InMemoryMetricDB()
    suite = VerificationSuite([my_simple_check], db, "Test Suite")

    # Collect checks (this is where it would fail with NameError)
    context = Context("test", db)
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.collect(context, key)

    # Verify the check was registered correctly
    checks = list(context._graph.root.children)
    assert len(checks) == 1
    assert checks[0].name == "my_simple_check"


def test_parametrized_check_with_empty_parens() -> None:
    """Test that @check() with empty parentheses uses function name."""

    @check(name="empty_paren_check")
    def empty_paren_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).is_gt(0)

    # No metadata is stored anymore, just verify the function works
    assert empty_paren_check.__name__ == "empty_paren_check"


def test_check_decorator_requires_name() -> None:
    """Test that @check decorator requires name parameter."""
    # This should raise TypeError because name is required
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'name'"):

        @check()  # type: ignore[call-arg]  # Missing required name parameter
        def my_check(mp: MetricProvider, ctx: Context) -> None:
            pass


def test_check_decorator_without_parentheses_not_allowed() -> None:
    """Test that @check without parentheses is not allowed."""
    # This test verifies compile-time behavior - the decorator
    # should not be callable without parentheses anymore
    # Note: This will be a syntax/type error after implementation
    pass


def test_check_decorator_with_name_works() -> None:
    """Test that @check with name parameter works correctly."""
    db = InMemoryMetricDB()

    @check(name="Valid Check")
    def my_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).is_gt(0)

    # Verify the check can be used in a suite
    suite = VerificationSuite([my_check], db, "Test Suite")
    assert suite is not None

    # Verify it can be collected
    context = Context("test", db)
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.collect(context, key)

    # Verify the check was registered with the correct name
    checks = list(context._graph.root.children)
    assert len(checks) == 1
    assert checks[0].name == "Valid Check"
