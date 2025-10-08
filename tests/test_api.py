"""Tests for API changes: removing listener pattern and assertion chaining."""

import datetime

import pytest
import sympy as sp
from dqx.api import AssertBuilder, Context, VerificationSuite, check, MetricProvider
from dqx.common import ResultKey
from dqx.orm.repositories import InMemoryMetricDB
from dqx.graph.nodes import AssertionNode


def test_assertion_node_is_immutable() -> None:
    """AssertionNode should be immutable after creation."""
    # Test that AssertionNode doesn't have setter methods
    node = AssertionNode(actual=sp.Symbol("x"))

    # These methods should not exist
    assert not hasattr(node, "set_label")
    assert not hasattr(node, "set_severity")
    assert not hasattr(node, "set_validator")

    # Fields can be set at construction but not modified after
    node_with_label = AssertionNode(actual=sp.Symbol("x"), label="test label")
    assert node_with_label.label == "test label"


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
    @check  # type: ignore[arg-type]
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

    @check  # type: ignore[arg-type]
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

    @check  # type: ignore[arg-type]
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        metric = sp.Symbol("x")

        # Multiple assertions on same metric (not chained)
        ctx.assert_that(metric).on(label="Greater than 40").is_gt(40)
        ctx.assert_that(metric).on(label="Less than 60").is_lt(60)

        # Verify we have 2 separate assertions
        check_node = ctx.current_check
        assert check_node is not None
        assert len(check_node.children) == 2

        # Verify labels were set correctly
        assert check_node.children[0].label == "Greater than 40"
        assert check_node.children[1].label == "Less than 60"

    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.collect(context, key=key)
