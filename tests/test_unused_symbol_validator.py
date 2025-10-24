"""Test UnusedSymbolValidator correctly handles dependency symbols."""

from dqx.api import Context, check
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.validator import SuiteValidator


def test_unused_validator_ignores_dependency_symbols() -> None:
    """Test that UnusedSymbolValidator ignores symbols with parents."""
    # GIVEN: A metric provider with extended metrics
    db = InMemoryMetricDB()
    mp = MetricProvider(db)

    # Create base metric and extended metric
    base = mp.maximum("tax")
    dod = mp.ext.day_over_day(base)

    # Create a graph with only dod being used
    root = RootNode("test_suite")

    @check(name="test_check")
    def test_check_fn(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(dod).where(name="dod > 0.9").is_gt(0.9)

    # Execute the check to build the graph
    context = Context(suite="test_suite", db=db)
    context._graph = Graph(root)
    test_check_fn(mp, context)

    graph = context._graph

    # WHEN: Running validation
    validator = SuiteValidator()
    report = validator.validate(graph, mp)

    # THEN: Should not report base or lag(1) as unused (they have parents)
    unused_warnings = [issue for issue in report.warnings if issue.rule == "unused_symbols"]
    unused_symbols_str = str(unused_warnings)

    # Base metric should not be reported as unused (it's the parent of dod)
    assert "maximum(tax)" not in unused_symbols_str
    # The lag(1) dependency should not be reported as unused (it has base as parent)
    assert "lag(1)" not in unused_symbols_str


def test_unused_validator_reports_truly_unused_symbols() -> None:
    """Test that UnusedSymbolValidator still reports truly unused symbols."""
    # GIVEN: A metric provider with multiple metrics
    db = InMemoryMetricDB()
    mp = MetricProvider(db)

    # Create multiple unrelated metrics
    tax = mp.maximum("tax")
    revenue = mp.sum("revenue")  # noqa: F841 - intentionally unused for test
    count = mp.num_rows()  # noqa: F841 - intentionally unused for test

    # Create a graph with only tax being used
    root = RootNode("test_suite")

    @check(name="test_check")
    def test_check_fn(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(tax).where(name="tax > 0").is_gt(0)

    # Execute the check to build the graph
    context = Context(suite="test_suite", db=db)
    context._graph = Graph(root)
    test_check_fn(mp, context)

    graph = context._graph

    # WHEN: Running validation
    validator = SuiteValidator()
    report = validator.validate(graph, mp)

    # THEN: Should report unused symbols
    unused_warnings = [issue for issue in report.warnings if issue.rule == "unused_symbols"]
    assert len(unused_warnings) > 0

    unused_str = str(unused_warnings)
    assert "sum(revenue)" in unused_str
    assert "num_rows()" in unused_str
    assert "maximum(tax)" not in unused_str  # This one is used


def test_unused_validator_with_mixed_symbols() -> None:
    """Test UnusedSymbolValidator with mix of base, extended, and dependency symbols."""
    # GIVEN: A complex metric hierarchy
    db = InMemoryMetricDB()
    mp = MetricProvider(db)

    # Create metrics
    tax = mp.maximum("tax")
    revenue = mp.sum("revenue")
    tax_dod = mp.ext.day_over_day(tax)
    revenue_wow = mp.ext.week_over_week(revenue)
    unused_metric = mp.minimum("cost")  # noqa: F841 - Not used anywhere (intentionally for test)

    # Create a graph using only extended metrics
    root = RootNode("test_suite")

    @check(name="test_check")
    def test_check_fn(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(tax_dod).where(name="tax_dod > 0.9").is_gt(0.9)
        ctx.assert_that(revenue_wow).where(name="revenue_wow > 0.8").is_gt(0.8)

    # Execute the check to build the graph
    context = Context(suite="test_suite", db=db)
    context._graph = Graph(root)
    test_check_fn(mp, context)

    graph = context._graph

    # WHEN: Running validation
    validator = SuiteValidator()
    report = validator.validate(graph, mp)

    # THEN: Only truly unused metrics should be reported
    unused_warnings = [issue for issue in report.warnings if issue.rule == "unused_symbols"]
    unused_str = str(unused_warnings)

    # Truly unused
    assert "minimum(cost)" in unused_str

    # These should NOT be reported as unused:
    # Base metrics (they are parents of used extended metrics)
    assert "maximum(tax)" not in unused_str
    assert "sum(revenue)" not in unused_str
    # Extended metrics (they are used)
    assert "day_over_day" not in unused_str
    assert "week_over_week" not in unused_str
    # Lag dependencies (they have parents)
    assert "lag(1)" not in unused_str
    assert "lag(7)" not in unused_str
