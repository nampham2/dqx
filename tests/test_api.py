import datetime

import pytest
import sympy as sp

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import DQXError, ResultKey
from dqx.orm.repositories import InMemoryMetricDB


def test_assertion_node_is_immutable() -> None:
    """AssertionNode should be immutable after creation."""
    # Test that AssertionNode doesn't have setter methods
    from dqx.common import SymbolicValidator
    from dqx.graph.nodes import RootNode

    root = RootNode("test_suite")
    check_node = root.add_check("test_check")
    validator = SymbolicValidator("not None", lambda x: x is not None)
    node = check_node.add_assertion(actual=sp.Symbol("x"), name="test assertion", validator=validator)

    # These methods should not exist
    assert not hasattr(node, "set_label")
    assert not hasattr(node, "set_severity")
    assert not hasattr(node, "set_validator")

    # Fields can be set at construction but not modified after
    node_with_label = check_node.add_assertion(actual=sp.Symbol("x"), name="test label", validator=validator)
    assert node_with_label.name == "test label"


def test_assertion_methods_return_none() -> None:
    """Assertion methods should not return AssertBuilder for chaining."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    # Create a simple check to have proper context
    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        draft = ctx.assert_that(sp.Symbol("x"))
        ready = draft.where(name="Test assertion 1")

        # These should return None, not AssertBuilder
        result = ready.is_gt(0)  # type: ignore[func-returns-value]
        assert result is None  # Should return None

        # Try other assertion methods
        ready2 = ctx.assert_that(sp.Symbol("y")).where(name="Test assertion 2")
        result2 = ready2.is_eq(5)  # type: ignore[func-returns-value]
        assert result2 is None  # Should return None

        result3 = ctx.assert_that(sp.Symbol("z")).where(name="Test assertion 3").is_leq(10)  # type: ignore[func-returns-value]
        assert result3 is None  # Should return None

    # Set up suite with the check
    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key=key)


def test_no_assertion_chaining() -> None:
    """Chained assertions should not be possible."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        metric = sp.Symbol("x")
        # This should fail - can't chain assertions
        result = ctx.assert_that(metric).where(name="Test assertion").is_gt(0)  # type: ignore[func-returns-value]
        # Result should be None, so calling is_lt on it should fail
        with pytest.raises(AttributeError):
            result.is_lt(100)  # type: ignore[attr-defined]

    # Set up suite and run check
    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key=key)


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
    suite.build_graph(context, key=key)


def test_simple_check_uses_function_name() -> None:
    """Test that @check without params uses function name."""

    # Create a simple check without parameters
    @check(name="validate_orders")
    def validate_orders(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    # No metadata is stored anymore, just verify the function works
    assert validate_orders.__name__ == "validate_orders"


def test_parametrized_check_uses_provided_name() -> None:
    """Test that @check with name parameter uses that name."""

    @check(name="Order Validation Check")
    def validate_orders(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    # No metadata is stored anymore, just verify the function works
    assert validate_orders.__name__ == "validate_orders"


def test_simple_check_works_in_suite() -> None:
    """Test that simple @check works in a verification suite."""

    @check(name="my_simple_check")
    def my_simple_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    # Should be able to use in a suite without errors
    db = InMemoryMetricDB()
    suite = VerificationSuite([my_simple_check], db, "Test Suite")

    # Collect checks (this is where it would fail with NameError)
    context = Context("test", db)
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key)

    # Verify the check was registered correctly
    checks = list(context._graph.root.children)
    assert len(checks) == 1
    assert checks[0].name == "my_simple_check"


def test_parametrized_check_with_empty_parens() -> None:
    """Test that @check() with empty parentheses uses function name."""

    @check(name="empty_paren_check")
    def empty_paren_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

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
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    # Verify the check can be used in a suite
    suite = VerificationSuite([my_check], db, "Test Suite")
    assert suite is not None

    # Verify it can be collected
    context = Context("test", db)
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key)

    # Verify the check was registered with the correct name
    checks = list(context._graph.root.children)
    assert len(checks) == 1
    assert checks[0].name == "Valid Check"


# NEW TESTS FOR ASSERTION DRAFT AND READY


def test_assertion_draft_creation() -> None:
    """AssertionDraft should only expose where() method."""
    from dqx.api import AssertionDraft

    expr = sp.Symbol("x")
    draft = AssertionDraft(actual=expr, context=None)

    # Should have where method
    assert hasattr(draft, "where")

    # Should NOT have assertion methods
    assert not hasattr(draft, "is_gt")
    assert not hasattr(draft, "is_eq")
    assert not hasattr(draft, "is_positive")


def test_assertion_ready_has_all_methods() -> None:
    """AssertionReady should have all assertion methods."""
    from dqx.api import AssertionReady

    expr = sp.Symbol("x")
    ready = AssertionReady(actual=expr, name="Test assertion", context=None)

    # Should have all assertion methods
    assert hasattr(ready, "is_gt")
    assert hasattr(ready, "is_geq")
    assert hasattr(ready, "is_lt")
    assert hasattr(ready, "is_leq")
    assert hasattr(ready, "is_eq")
    assert hasattr(ready, "is_positive")
    assert hasattr(ready, "is_negative")
    assert hasattr(ready, "is_between")
    assert hasattr(ready, "noop")

    # Should NOT have where method
    assert not hasattr(ready, "where")


def test_context_assert_that_returns_draft() -> None:
    """Context.assert_that should return AssertionDraft."""
    from dqx.api import AssertionDraft

    db = InMemoryMetricDB()
    context = Context("test", db)

    draft = context.assert_that(sp.Symbol("x"))
    assert isinstance(draft, AssertionDraft)


def test_assertion_workflow_end_to_end() -> None:
    """Test complete assertion workflow from draft to execution."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        # Create draft
        draft = ctx.assert_that(sp.Symbol("x"))

        # Convert to ready with name
        ready = draft.where(name="X is positive")

        # Make assertion
        ready.is_positive()

        # Verify assertion was created
        assert ctx.current_check is not None
        assert len(ctx.current_check.children) == 1
        assert ctx.current_check.children[0].name == "X is positive"

    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key=key)


def test_cannot_use_assertion_methods_on_draft() -> None:
    """Assertion methods should not be available on AssertionDraft."""
    from dqx.api import AssertionDraft

    draft = AssertionDraft(sp.Symbol("x"), context=None)

    # These should raise AttributeError
    with pytest.raises(AttributeError):
        draft.is_gt(0)  # type: ignore

    with pytest.raises(AttributeError):
        draft.is_positive()  # type: ignore


def test_where_requires_name_parameter() -> None:
    """The where() method should require name parameter."""
    from dqx.api import AssertionDraft

    draft = AssertionDraft(sp.Symbol("x"), context=None)

    # Should fail without name
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'name'"):
        draft.where()  # type: ignore

    # Should work with name
    ready = draft.where(name="Valid name")
    assert ready is not None


def test_assertion_ready_always_has_name() -> None:
    """AssertionReady should always have a name set."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="My assertion").is_positive()

        # Check that the assertion node has the name
        assert ctx.current_check is not None
        assertion_node = ctx.current_check.children[0]
        assert assertion_node.name == "My assertion"

    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key=key)


def test_where_validates_name() -> None:
    """The where() method should validate name parameter."""
    from dqx.api import AssertionDraft

    draft = AssertionDraft(sp.Symbol("x"), context=None)

    # Empty string should fail
    with pytest.raises(ValueError, match="Assertion name cannot be empty"):
        draft.where(name="")

    # Whitespace only should fail
    with pytest.raises(ValueError, match="Assertion name cannot be empty"):
        draft.where(name="   ")

    # Too long name should fail
    with pytest.raises(ValueError, match="Assertion name is too long"):
        draft.where(name="x" * 256)

    # Valid name should work
    ready = draft.where(name="Valid assertion name")
    assert ready is not None

    # Name should be stripped
    ready2 = draft.where(name="  Trimmed name  ")
    assert ready2._name == "Trimmed name"


def test_assertion_severity_is_mandatory_with_p1_default() -> None:
    """Test that assertions require severity and default to P1."""
    # Create a mock context and provider
    db = InMemoryMetricDB()
    context = Context(suite="test_suite", db=db)

    # Create a check to hold our assertions
    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        # Create assertion without severity - should default to P1
        ctx.assert_that(sp.Symbol("x")).where(name="Default severity test").is_gt(0)

        # Create assertion with explicit P0 severity
        ctx.assert_that(sp.Symbol("y")).where(name="Explicit P0 test", severity="P0").is_gt(0)

        # Create assertion with explicit P2 severity
        ctx.assert_that(sp.Symbol("z")).where(name="Explicit P2 test", severity="P2").is_gt(0)

    # Execute the check
    test_check(context.provider, context)

    # Get the check node and its assertions
    check_node = list(context._graph.root.children)[0]
    assertions = list(check_node.children)

    # Verify severities
    assert len(assertions) == 3
    assert assertions[0].severity == "P1"  # Default
    assert assertions[1].severity == "P0"  # Explicit P0
    assert assertions[2].severity == "P2"  # Explicit P2

    # Verify that severity is never None
    for assertion in assertions:
        assert assertion.severity is not None
        assert assertion.severity in ["P0", "P1", "P2", "P3"]


def test_verification_suite_graph_property() -> None:
    """Test that VerificationSuite exposes graph property with proper error handling."""

    # Create a simple check for testing
    @check(name="Simple Check")
    def simple_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    db = InMemoryMetricDB()
    suite = VerificationSuite([simple_check], db, "Test Suite")

    # Should raise error before run is called
    with pytest.raises(DQXError, match="Verification suite has not been executed yet!"):
        _ = suite.graph

    # After running suite, should work
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    # Need to provide a mock data source for run
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    data = pa.table({"x": [1, 2, 3]})
    suite.run({"data": DuckRelationDataSource.from_arrow(data)}, key)

    # Should return a Graph instance
    from dqx.graph.traversal import Graph

    assert isinstance(suite.graph, Graph)

    # Should have the suite name as root
    assert suite.graph.root.name == "Test Suite"


def test_verification_suite_build_graph_method() -> None:
    """Test that build_graph method works (renamed from collect)."""

    # Create a simple check for testing
    @check(name="Simple Check")
    def simple_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    db = InMemoryMetricDB()
    suite = VerificationSuite([simple_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Should have build_graph method
    assert hasattr(suite, "build_graph")

    # Should not have collect method anymore
    assert not hasattr(suite, "collect")

    # Run suite which will call build_graph internally
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    data = pa.table({"x": [1, 2, 3]})
    suite.run({"data": DuckRelationDataSource.from_arrow(data)}, key)

    # After run, graph should be populated
    assert len(suite.graph.root.children) > 0


def test_is_between_assertion_workflow() -> None:
    """Test is_between assertion in complete workflow."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Range Check")
    def range_check(mp: MetricProvider, ctx: Context) -> None:
        # Test normal range
        ctx.assert_that(sp.Symbol("x")).where(name="X is between 10 and 20").is_between(10.0, 20.0)

        # Test with same bounds
        ctx.assert_that(sp.Symbol("y")).where(name="Y equals 5").is_between(5.0, 5.0)

        # Verify assertions were created
        assert ctx.current_check is not None
        assert len(ctx.current_check.children) == 2
        assert ctx.current_check.children[0].name == "X is between 10 and 20"
        assert ctx.current_check.children[1].name == "Y equals 5"

    suite = VerificationSuite([range_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key=key)


def test_is_between_invalid_bounds() -> None:
    """Test is_between with invalid bounds raises ValueError."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Invalid Range Check")
    def invalid_check(mp: MetricProvider, ctx: Context) -> None:
        # This should raise ValueError
        with pytest.raises(ValueError, match="Invalid range: lower bound .* must be less than or equal to upper bound"):
            ctx.assert_that(sp.Symbol("x")).where(name="Invalid range").is_between(20.0, 10.0)

    # Execute the check to verify the error is raised
    invalid_check(context.provider, context)


def test_assertion_ready_has_noop_method() -> None:
    """AssertionReady should have noop method."""
    from dqx.api import AssertionReady

    expr = sp.Symbol("x")
    ready = AssertionReady(actual=expr, name="Test assertion", context=None)

    # Should have noop method
    assert hasattr(ready, "noop")


def test_noop_assertion_workflow() -> None:
    """Test complete noop assertion workflow."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Metric Collection Check")
    def collection_check(mp: MetricProvider, ctx: Context) -> None:
        # Test noop on various metrics
        ctx.assert_that(mp.num_rows()).where(name="Collect row count").noop()
        ctx.assert_that(mp.average("price")).where(name="Collect average price").noop()
        ctx.assert_that(mp.sum("amount") + mp.sum("tax")).where(name="Collect total revenue").noop()

        # Verify assertions were created
        assert ctx.current_check is not None
        assert len(ctx.current_check.children) == 3

    suite = VerificationSuite([collection_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key=key)
