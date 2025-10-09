# tests/test_assertion_rename.py
import pytest
import sympy as sp

from dqx.api import Context, check
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_assert_builder_where_method():
    """Test that AssertBuilder has where() method instead of on()."""
    db = InMemoryMetricDB()
    ctx = Context("test_suite", db)

    # Create a simple expression
    expr = sp.Symbol("x")

    # This should work with new API
    assert_builder = ctx.assert_that(expr)

    # Should have where method
    assert hasattr(assert_builder, "where")

    # Should NOT have on method
    assert not hasattr(assert_builder, "on")

    # Test method chaining
    result = assert_builder.where(name="Test assertion")
    assert result is assert_builder  # Should return self


def test_where_method_parameters():
    """Test that where() accepts name parameter instead of label."""
    db = InMemoryMetricDB()
    ctx = Context("test_suite", db)

    # Should accept name parameter
    ctx.assert_that(sp.Symbol("x")).where(name="Test name").is_eq(42)

    # Should NOT accept label parameter
    with pytest.raises(TypeError, match="unexpected keyword argument 'label'"):
        ctx.assert_that(sp.Symbol("x")).where(label="Test label").is_eq(42)


def test_check_decorator_with_name():
    """Test that @check decorator accepts name instead of label."""

    # Define a check with name parameter
    @check(name="My validation check", tags=["test"])
    def my_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    # Verify metadata is stored correctly
    assert hasattr(my_check, "_check_metadata")
    assert my_check._check_metadata["display_name"] == "My validation check"
    assert my_check._check_metadata["name"] == "my_check"  # Function name


def test_check_decorator_without_name():
    """Test that @check decorator uses function name when name not provided."""

    @check(tags=["test"])
    def validate_something(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).is_gt(0)

    # Should use function name
    assert validate_something._check_metadata["name"] == "validate_something"
    assert validate_something._check_metadata["display_name"] is None
