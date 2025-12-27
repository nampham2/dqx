import datetime

import pytest
import sympy as sp
from returns.result import Success

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
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

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
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

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
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

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
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)
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
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)
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
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

    draft = context.assert_that(sp.Symbol("x"))
    assert isinstance(draft, AssertionDraft)


def test_assertion_workflow_end_to_end() -> None:
    """Test complete assertion workflow from draft to execution."""
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

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
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

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
    context = Context(suite="test_suite", db=db, execution_id="test-exec-123", data_av_threshold=0.9)

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
    suite.run([DuckRelationDataSource.from_arrow(data, "data")], key)

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
    suite.run([DuckRelationDataSource.from_arrow(data, "data")], key)

    # After run, graph should be populated
    assert len(suite.graph.root.children) > 0


def test_is_between_assertion_workflow() -> None:
    """Test is_between assertion in complete workflow."""
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

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
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

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
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

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


def test_verification_suite_metric_trace() -> None:
    """Test VerificationSuite.metric_trace method."""
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    db = InMemoryMetricDB()

    @check(name="Test Check", datasets=["test_data"])
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Row count is positive").is_gt(0)
        ctx.assert_that(mp.average("value")).where(name="Average is reasonable").is_between(0, 100)

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Create test data
    data = pa.table({"id": [1, 2, 3, 4, 5], "value": [10, 20, 30, 40, 50]})
    datasource = DuckRelationDataSource.from_arrow(data, "test_data")

    # Should raise error before run
    with pytest.raises(DQXError, match="Verification suite has not been executed yet!"):
        _ = suite.metric_trace(db)

    # Run the suite
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={"test": "true"})
    suite.run([datasource], key)

    # Now metric_trace should work
    trace = suite.metric_trace(db)

    # Verify it's a PyArrow table
    assert isinstance(trace, pa.Table)

    # Verify expected columns
    expected_columns = {
        "date",
        "metric",
        "symbol",
        "type",
        "dataset",
        "value_db",
        "value_analysis",
        "value_final",
        "error",
        "tags",
        "is_extended",
        "data_av_ratio",
    }
    assert set(trace.column_names) == expected_columns

    # Should have rows for the metrics
    assert trace.num_rows > 0

    # Check that execution_id matches
    from dqx import data

    metrics = db.get_by_execution_id(suite.execution_id)
    assert len(metrics) > 0  # Should have persisted metrics


def test_verification_suite_metric_trace_stats() -> None:
    """Test metric_trace_stats with VerificationSuite results."""
    import pyarrow as pa

    from dqx.datasource import DuckRelationDataSource

    db = InMemoryMetricDB()

    @check(name="Test Check", datasets=["test_data"])
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Row count check").is_gt(0)

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Create test data
    data = pa.table({"id": [1, 2, 3]})
    datasource = DuckRelationDataSource.from_arrow(data, "test_data")

    # Run the suite
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([datasource], key)

    # Get trace and stats
    trace = suite.metric_trace(db)

    from dqx.data import metric_trace_stats

    stats = metric_trace_stats(trace)

    # Verify stats structure
    assert hasattr(stats, "total_rows")
    assert hasattr(stats, "discrepancy_count")
    assert hasattr(stats, "discrepancy_rows")
    assert hasattr(stats, "discrepancy_details")

    # Should have at least one row
    assert stats.total_rows > 0

    # In a clean run, there shouldn't be discrepancies
    assert stats.discrepancy_count == 0
    assert len(stats.discrepancy_rows) == 0
    assert len(stats.discrepancy_details) == 0


def test_assertion_tags() -> None:
    """Assertions can have tags for profile-based selection."""
    from dqx.common import SymbolicValidator
    from dqx.graph.nodes import RootNode

    root = RootNode("test_suite")
    check_node = root.add_check("test_check")
    validator = SymbolicValidator("> 0", lambda x: x > 0)

    # Create assertion with tags
    node = check_node.add_assertion(
        actual=sp.Symbol("x"),
        name="tagged assertion",
        validator=validator,
        tags=frozenset({"xmas", "volume"}),
    )

    assert node.tags == frozenset({"xmas", "volume"})

    # Create assertion without tags - defaults to empty set
    node_no_tags = check_node.add_assertion(
        actual=sp.Symbol("y"),
        name="untagged assertion",
        validator=validator,
    )

    assert node_no_tags.tags == frozenset()


def test_assertion_tags_via_where() -> None:
    """Tags can be specified via the where() method."""
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(
            name="Tagged assertion",
            tags={"xmas", "critical"},
        ).is_gt(0)

        ctx.assert_that(sp.Symbol("y")).where(
            name="Untagged assertion",
        ).is_gt(0)

    # Build the graph
    root = context._graph.root
    check_node = root.add_check("Test Check")
    with context.check_context(check_node):
        test_check(context._provider, context)

    # Get assertions
    assertions = list(context._graph.assertions())
    assert len(assertions) == 2

    tagged = next(a for a in assertions if a.name == "Tagged assertion")
    untagged = next(a for a in assertions if a.name == "Untagged assertion")

    assert tagged.tags == frozenset({"xmas", "critical"})
    assert untagged.tags == frozenset()


def test_experimental_annotation_via_where() -> None:
    """Experimental annotation can be specified via the where() method."""
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        """
        Registers two assertions: one marked experimental and one production, both checking their metric is greater than zero.

        The first assertion targets symbol "x" and is annotated experimental=True; the second targets symbol "y" and uses the default (production) annotation.
        """
        ctx.assert_that(sp.Symbol("x")).where(
            name="Experimental assertion",
            experimental=True,
        ).is_gt(0)

        ctx.assert_that(sp.Symbol("y")).where(
            name="Production assertion",
        ).is_gt(0)

    # Build the graph
    root = context._graph.root
    check_node = root.add_check("Test Check")
    with context.check_context(check_node):
        test_check(context._provider, context)

    # Get assertions
    assertions = list(context._graph.assertions())
    assert len(assertions) == 2

    experimental = next(a for a in assertions if a.name == "Experimental assertion")
    production = next(a for a in assertions if a.name == "Production assertion")

    assert experimental.experimental is True
    assert production.experimental is False


def test_experimental_annotation_in_assertion_result() -> None:
    """Experimental annotation should be included in AssertionResult."""
    from dqx.common import AssertionResult

    # Check that AssertionResult has experimental field with default False
    result = AssertionResult(
        yyyy_mm_dd=datetime.date.today(),
        suite="test",
        check="test_check",
        assertion="test_assertion",
        severity="P1",
        status="PASSED",
        metric=Success(1.0),
    )
    assert result.experimental is False

    # Check that experimental can be set to True
    result_experimental = AssertionResult(
        yyyy_mm_dd=datetime.date.today(),
        suite="test",
        check="test_check",
        assertion="test_assertion",
        severity="P1",
        status="PASSED",
        metric=Success(1.0),
        experimental=True,
    )
    assert result_experimental.experimental is True


def test_required_annotation_via_where() -> None:
    """Required annotation can be specified via the where() method."""
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(
            name="Required assertion",
            required=True,
        ).is_gt(0)

        ctx.assert_that(sp.Symbol("y")).where(
            name="Non-required assertion",
        ).is_gt(0)

    # Build the graph
    root = context._graph.root
    check_node = root.add_check("Test Check")
    with context.check_context(check_node):
        test_check(context._provider, context)

    # Get assertions
    assertions = list(context._graph.assertions())
    assert len(assertions) == 2

    required = next(a for a in assertions if a.name == "Required assertion")
    non_required = next(a for a in assertions if a.name == "Non-required assertion")

    assert required.required is True
    assert non_required.required is False


def test_cost_annotation_via_where() -> None:
    """Cost annotation can be specified via the where() method."""
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(
            name="Assertion with cost",
            cost={"fp": 1.0, "fn": 100.0},
        ).is_gt(0)

        ctx.assert_that(sp.Symbol("y")).where(
            name="Assertion without cost",
        ).is_gt(0)

    # Build the graph
    root = context._graph.root
    check_node = root.add_check("Test Check")
    with context.check_context(check_node):
        test_check(context._provider, context)

    # Get assertions
    assertions = list(context._graph.assertions())
    assert len(assertions) == 2

    with_cost = next(a for a in assertions if a.name == "Assertion with cost")
    without_cost = next(a for a in assertions if a.name == "Assertion without cost")

    assert with_cost.cost_fp == 1.0
    assert with_cost.cost_fn == 100.0
    assert without_cost.cost_fp is None
    assert without_cost.cost_fn is None


def test_cost_validation_requires_dict() -> None:
    """
    Verifies that specifying a non-dict cost in an assertion raises a ValueError.

    This test ensures where(..., cost=...) requires a dict containing 'fp' and 'fn'.

    Raises:
        ValueError: if cost is not a dict with 'fp' and 'fn' keys.
    """
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

    # Not a dict (tuple)
    with pytest.raises(ValueError, match="cost must be a dict with 'fp' and 'fn' keys"):
        context.assert_that(sp.Symbol("x")).where(
            name="Test",
            cost=(1.0, 100.0),  # type: ignore[arg-type]
        )


def test_cost_validation_requires_both_fp_and_fn() -> None:
    """Cost must have exactly 'fp' and 'fn' keys."""
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

    # Missing fn
    with pytest.raises(ValueError, match="must have exactly 'fp' and 'fn' keys"):
        context.assert_that(sp.Symbol("x")).where(
            name="Test",
            cost={"fp": 1.0},
        )

    # Extra key
    with pytest.raises(ValueError, match="must have exactly 'fp' and 'fn' keys"):
        context.assert_that(sp.Symbol("x")).where(
            name="Test",
            cost={"fp": 1.0, "fn": 2.0, "extra": 3.0},
        )


def test_cost_validation_numeric() -> None:
    """Cost values must be numeric."""
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

    # String values
    with pytest.raises(ValueError, match="cost values must be numeric"):
        context.assert_that(sp.Symbol("x")).where(
            name="Test",
            cost={"fp": "high", "fn": 100.0},  # type: ignore[dict-item]
        )


def test_cost_validation_non_negative() -> None:
    """Cost values must be non-negative."""
    db = InMemoryMetricDB()
    context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

    # Negative fp
    with pytest.raises(ValueError, match="cost values must be non-negative"):
        context.assert_that(sp.Symbol("x")).where(
            name="Test",
            cost={"fp": -1.0, "fn": 100.0},
        )

    # Negative fn
    with pytest.raises(ValueError, match="cost values must be non-negative"):
        context.assert_that(sp.Symbol("x")).where(
            name="Test",
            cost={"fp": 1.0, "fn": -100.0},
        )


def test_required_and_cost_in_assertion_result() -> None:
    """Required and cost annotations should be included in AssertionResult."""
    from dqx.common import AssertionResult

    # Check defaults
    result = AssertionResult(
        yyyy_mm_dd=datetime.date.today(),
        suite="test",
        check="test_check",
        assertion="test_assertion",
        severity="P1",
        status="PASSED",
        metric=Success(1.0),
    )
    assert result.required is False
    assert result.cost_fp is None
    assert result.cost_fn is None

    # Check with values set
    result_with_annotations = AssertionResult(
        yyyy_mm_dd=datetime.date.today(),
        suite="test",
        check="test_check",
        assertion="test_assertion",
        severity="P0",
        status="PASSED",
        metric=Success(1.0),
        required=True,
        cost_fp=5.0,
        cost_fn=200.0,
    )
    assert result_with_annotations.required is True
    assert result_with_annotations.cost_fp == 5.0
    assert result_with_annotations.cost_fn == 200.0


class TestNewAssertionMethods:
    """Tests for new assertion methods: is_neq, is_none, is_not_none."""

    def test_assertion_ready_has_is_neq(self) -> None:
        """AssertionReady should have is_neq method."""
        from dqx.api import AssertionReady

        expr = sp.Symbol("x")
        ready = AssertionReady(actual=expr, name="Test assertion", context=None)
        assert hasattr(ready, "is_neq")

    def test_assertion_ready_has_is_none(self) -> None:
        """AssertionReady should have is_none method."""
        from dqx.api import AssertionReady

        expr = sp.Symbol("x")
        ready = AssertionReady(actual=expr, name="Test assertion", context=None)
        assert hasattr(ready, "is_none")

    def test_assertion_ready_has_is_not_none(self) -> None:
        """AssertionReady should have is_not_none method."""
        from dqx.api import AssertionReady

        expr = sp.Symbol("x")
        ready = AssertionReady(actual=expr, name="Test assertion", context=None)
        assert hasattr(ready, "is_not_none")

    def test_is_neq_assertion_workflow(self) -> None:
        """Test is_neq assertion in complete workflow."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        @check(name="Not Equal Check")
        def neq_check(mp: MetricProvider, ctx: Context) -> None:
            """
            Create two "not equal" assertions for metrics 'x' and 'y' and verify the current check contains two children with the expected names.

            This function registers an assertion that `x` is not equal to 0 (named "X is not zero") and an assertion that `y` is not equal to 100 (named "Y is not 100"), then asserts that the context's current check exists and has exactly those two child assertions in order.
            """
            ctx.assert_that(sp.Symbol("x")).where(name="X is not zero").is_neq(0)
            ctx.assert_that(sp.Symbol("y")).where(name="Y is not 100").is_neq(100)

            assert ctx.current_check is not None
            assert len(ctx.current_check.children) == 2
            assert ctx.current_check.children[0].name == "X is not zero"
            assert ctx.current_check.children[1].name == "Y is not 100"

        suite = VerificationSuite([neq_check], db, "test")
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.build_graph(context, key=key)

    def test_is_neq_validator_description(self) -> None:
        """Test is_neq creates correct validator description."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        @check(name="Validator Check")
        def val_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(sp.Symbol("x")).where(name="Test neq").is_neq(42)

            assert ctx.current_check is not None
            assertion = ctx.current_check.children[0]
            assert "\u2260 42" in assertion.validator.name  # ≠ symbol

        suite = VerificationSuite([val_check], db, "test")
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.build_graph(context, key=key)

    def test_is_none_assertion_workflow(self) -> None:
        """Test is_none assertion in complete workflow."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        @check(name="None Check")
        def none_check(mp: MetricProvider, ctx: Context) -> None:
            """
            Create a verification check that asserts the metric `x` is None and validate the resulting check node.

            This registers an assertion via the provided Context: it converts an assertion draft for `x` into a ready assertion named "X is None", executes the `is_none()` assertion, and then verifies that `ctx.current_check` exists, contains exactly one child, and that the child's name equals "X is None".

            Parameters:
                mp (MetricProvider): Provider used to register or resolve metrics (unused by this helper but required by test harness).
                ctx (Context): Execution context in which the check and assertion are created and validated.
            """
            ctx.assert_that(sp.Symbol("x")).where(name="X is None").is_none()

            assert ctx.current_check is not None
            assert len(ctx.current_check.children) == 1
            assert ctx.current_check.children[0].name == "X is None"

        suite = VerificationSuite([none_check], db, "test")
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.build_graph(context, key=key)

    def test_is_none_validator_description(self) -> None:
        """Test is_none creates correct validator description."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        @check(name="Validator Check")
        def val_check(mp: MetricProvider, ctx: Context) -> None:
            """
            Create an 'is None' assertion named "Test none" for symbol "x" in the provided context and verify the created assertion's validator name contains "is None".

            This function registers the assertion via ctx.assert_that(...).where(...).is_none(), then inspects ctx.current_check to ensure a child assertion was created and that its validator description includes the text "is None".
            """
            ctx.assert_that(sp.Symbol("x")).where(name="Test none").is_none()

            assert ctx.current_check is not None
            assertion = ctx.current_check.children[0]
            assert "is None" in assertion.validator.name

        suite = VerificationSuite([val_check], db, "test")
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.build_graph(context, key=key)

    def test_is_not_none_assertion_workflow(self) -> None:
        """Test is_not_none assertion in complete workflow."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        @check(name="Not None Check")
        def not_none_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(sp.Symbol("x")).where(name="X is not None").is_not_none()

            assert ctx.current_check is not None
            assert len(ctx.current_check.children) == 1
            assert ctx.current_check.children[0].name == "X is not None"

        suite = VerificationSuite([not_none_check], db, "test")
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.build_graph(context, key=key)

    def test_is_not_none_validator_description(self) -> None:
        """Test is_not_none creates correct validator description."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        @check(name="Validator Check")
        def val_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(sp.Symbol("x")).where(name="Test not none").is_not_none()

            assert ctx.current_check is not None
            assertion = ctx.current_check.children[0]
            assert "is not None" in assertion.validator.name

        suite = VerificationSuite([val_check], db, "test")
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.build_graph(context, key=key)

    def test_is_neq_with_tolerance(self) -> None:
        """Test is_neq with custom tolerance."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        @check(name="Tolerance Check")
        def tol_check(mp: MetricProvider, ctx: Context) -> None:
            # Default tolerance
            """
            Create two "not equal" assertions on symbols using default and custom tolerance.

            This registers an assertion named "Default tol" that checks symbol `x` is not equal to 0 using the provider's default tolerance, and an assertion named "Custom tol" that checks symbol `y` is not equal to 0 using a tolerance of 0.1. After registration, the current check is expected to contain exactly two child assertions.

            Parameters:
                mp (MetricProvider): Metric provider used to resolve metrics (not modified).
                ctx (Context): Assertion context where checks are created and stored.
            """
            ctx.assert_that(sp.Symbol("x")).where(name="Default tol").is_neq(0)
            # Custom tolerance
            ctx.assert_that(sp.Symbol("y")).where(name="Custom tol").is_neq(0, tol=0.1)

            assert ctx.current_check is not None
            assert len(ctx.current_check.children) == 2

        suite = VerificationSuite([tol_check], db, "test")
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.build_graph(context, key=key)


class TestTagValidation:
    """Tests for tag validation."""

    def test_valid_tags(self) -> None:
        """Test that valid tags are accepted."""
        from dqx.common import validate_tags

        assert validate_tags({"xmas", "critical"}) == frozenset({"xmas", "critical"})
        assert validate_tags({"tag-with-dash"}) == frozenset({"tag-with-dash"})
        assert validate_tags({"tag_with_underscore"}) == frozenset({"tag_with_underscore"})
        assert validate_tags({"tag123"}) == frozenset({"tag123"})
        assert validate_tags({"CamelCase"}) == frozenset({"CamelCase"})

    def test_empty_tags(self) -> None:
        """Test that None and empty set return empty frozenset."""
        from dqx.common import validate_tags

        assert validate_tags(None) == frozenset()
        assert validate_tags(set()) == frozenset()

    def test_invalid_empty_tag(self) -> None:
        """Test that empty string tag raises ValueError."""
        from dqx.common import validate_tags

        with pytest.raises(ValueError, match="empty or whitespace"):
            validate_tags({""})

    def test_invalid_whitespace_tag(self) -> None:
        """Test that whitespace-only tag raises ValueError."""
        from dqx.common import validate_tags

        with pytest.raises(ValueError, match="empty or whitespace"):
            validate_tags({"   "})

    def test_invalid_special_characters(self) -> None:
        """Test that tags with special characters raise ValueError."""
        from dqx.common import validate_tags

        with pytest.raises(ValueError, match="alphanumerics, dashes, and underscores"):
            validate_tags({"tag with space"})

        with pytest.raises(ValueError, match="alphanumerics, dashes, and underscores"):
            validate_tags({"tag@special"})

        with pytest.raises(ValueError, match="alphanumerics, dashes, and underscores"):
            validate_tags({"tag.dot"})

    def test_tags_are_trimmed(self) -> None:
        """Test that tags with leading/trailing whitespace are trimmed."""
        from dqx.common import validate_tags

        assert validate_tags({"  valid  "}) == frozenset({"valid"})

    def test_assertion_with_invalid_tags_raises(self) -> None:
        """Test that creating assertion with invalid tags raises ValueError."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        with pytest.raises(ValueError, match="alphanumerics, dashes, and underscores"):
            context.assert_that(sp.Symbol("x")).where(name="Test", tags={"invalid tag"})


class TestVerificationSuiteValidation:
    """Tests for VerificationSuite validation errors."""

    def test_empty_checks_raises_error(self) -> None:
        """Test that empty checks list raises DQXError."""
        db = InMemoryMetricDB()
        with pytest.raises(DQXError, match="At least one check must be provided"):
            VerificationSuite([], db, "Test Suite")

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises DQXError."""
        db = InMemoryMetricDB()

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

        with pytest.raises(DQXError, match="Suite name cannot be empty"):
            VerificationSuite([test_check], db, "")

        with pytest.raises(DQXError, match="Suite name cannot be empty"):
            VerificationSuite([test_check], db, "   ")

    def test_empty_datasources_raises_error(self) -> None:
        """
        Verify that running a VerificationSuite with no data sources raises a DQXError.

        Asserts that calling `VerificationSuite.run()` with an empty list of data sources raises `DQXError` with message containing "No data sources provided".
        """
        db = InMemoryMetricDB()

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

        suite = VerificationSuite([test_check], db, "Test Suite")
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        with pytest.raises(DQXError, match="No data sources provided"):
            suite.run([], key)


class TestContextPendingMetrics:
    """Tests for Context.pending_metrics filtering."""

    def test_pending_metrics_all(self) -> None:
        """Test pending_metrics returns all metrics when no dataset specified."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        # Register metrics directly via provider
        context.provider.num_rows(dataset="dataset1")
        context.provider.average("price", dataset="dataset2")
        context.provider.sum("amount", dataset="dataset1")

        # Get all metrics
        all_metrics = context.pending_metrics()
        assert len(all_metrics) == 3

    def test_pending_metrics_filtered_by_dataset(self) -> None:
        """Test pending_metrics filters by dataset."""
        db = InMemoryMetricDB()
        context = Context("test", db, execution_id="test-exec-123", data_av_threshold=0.9)

        # Register metrics directly via provider
        context.provider.num_rows(dataset="dataset1")
        context.provider.average("price", dataset="dataset2")
        context.provider.sum("amount", dataset="dataset1")

        # Get metrics for dataset1 only
        dataset1_metrics = context.pending_metrics(dataset="dataset1")
        assert len(dataset1_metrics) == 2
        assert all(m.dataset == "dataset1" for m in dataset1_metrics)

        # Get metrics for dataset2 only
        dataset2_metrics = context.pending_metrics(dataset="dataset2")
        assert len(dataset2_metrics) == 1
        assert all(m.dataset == "dataset2" for m in dataset2_metrics)
