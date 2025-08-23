import sympy as sp
import datetime as dt
from rich.console import Console
from rich.tree import Tree
from unittest.mock import MagicMock

from typing import Optional
from returns.result import Success
import pytest

from dqx import graph
from dqx.specs import MetricSpec
from dqx.common import ResultKey, ResultKeyProvider
from dqx.ops import Op
from dqx.graph import AssertionNode, SymbolNode, CheckNode, RootNode


# Helper classes
class MockOp(Op[float]):
    def __init__(self, name: str):
        self._name = name
        self._prefix = "mock"
        self._value: Optional[float] = None
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def prefix(self) -> str:
        return self._prefix
    
    def value(self) -> float:
        if self._value is None:
            raise RuntimeError("MockOp has no value")
        return self._value
    
    def assign(self, value: float) -> None:
        self._value = value
    
    def clear(self) -> None:
        self._value = None


class MockKeyProvider(ResultKeyProvider):
    def create(self, key: ResultKey) -> ResultKey:
        return key


# Helper functions
def walker(node: graph.Node) -> None:
    print(node.inspect_str())


# Original test from test_graph.py
def test_graph() -> None:
    root = graph.RootNode("Some checks")
    root.add_child(check_1:= graph.CheckNode("check_1"))
    check_1.add_child(graph.AssertionNode("assertion_1"))

    print("\n")
    tree = root.inspect()
    Console().print(tree)


# Tests from test_graph_spacing.py
def test_analyzer_node_spacing() -> None:
    """Test that analyzer nodes don't have extra spacing between them."""
    # Create a graph structure with analyzer nodes
    root = graph.RootNode("Test Suite")
    check = graph.CheckNode("test_check", datasets=["test_ds"])
    root.add_child(check)
    
    # Create a symbol with metrics and analyzers
    symbol = graph.SymbolNode(
        name="test_symbol",
        symbol=sp.Symbol("x"),
        fn=lambda key: Success(1.0),
        datasets=["test_ds"]
    )
    check.add_child(symbol)
    
    # Create metric nodes with analyzer children
    metric_spec1 = MagicMock(spec=MetricSpec)
    metric_spec1.name = "metric1"
    metric_spec2 = MagicMock(spec=MetricSpec)
    metric_spec2.name = "metric2"
    
    key_provider = MockKeyProvider()
    nominal_key = ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    
    metric1 = graph.MetricNode(metric_spec1, key_provider, nominal_key)
    metric2 = graph.MetricNode(metric_spec2, key_provider, nominal_key)
    
    symbol.add_child(metric1)
    symbol.add_child(metric2)
    
    # Add analyzer nodes to metrics
    analyzer1 = graph.AnalyzerNode(MockOp("analyzer1"))
    analyzer2 = graph.AnalyzerNode(MockOp("analyzer2"))
    analyzer3 = graph.AnalyzerNode(MockOp("analyzer3"))
    analyzer4 = graph.AnalyzerNode(MockOp("analyzer4"))
    
    metric1.add_child(analyzer1)
    metric1.add_child(analyzer2)
    metric2.add_child(analyzer3)
    metric2.add_child(analyzer4)
    
    # Inspect the tree
    tree = root.inspect()
    
    # Convert tree to string for checking
    console = Console(force_terminal=False, width=200)
    with console.capture() as capture:
        console.print(tree)
    
    output = capture.get()
    
    # Check that there are no double newlines between analyzer nodes
    # Double newlines would appear as '\n\n' in the output
    assert '\n\n' not in output, "Found extra spacing (double newlines) between analyzer nodes"
    
    # Also check the tree labels directly
    def check_tree_labels(tree: Tree, path: str = "root") -> None:
        label_str = str(tree.label)
        # Metric nodes should not have trailing newlines
        if "metric" in label_str.lower() and label_str.endswith("\n"):
            raise AssertionError(f"Metric node at {path} has trailing newline: {repr(label_str)}")
        
        for i, child in enumerate(tree.children):
            check_tree_labels(child, f"{path}/child[{i}]")
    
    check_tree_labels(tree)


def test_graph_display_structure() -> None:
    """Test the overall structure of graph display."""
    root = graph.RootNode("Test Suite")
    check = graph.CheckNode("check1", label="Check One")
    root.add_child(check)
    
    # Add assertion with validator
    from dqx.common import SymbolicValidator
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + sp.Symbol("y"),
        label="Test assertion",
        validator=SymbolicValidator(name="> 0", fn=lambda x: x > 0)
    )
    check.add_child(assertion)
    
    # Add symbol with metric
    symbol = graph.SymbolNode(
        name="symbol1",
        symbol=sp.Symbol("x"),
        fn=lambda key: Success(1.0),
        datasets=[]
    )
    check.add_child(symbol)
    
    metric_spec = MagicMock(spec=MetricSpec)
    metric_spec.name = "test_metric"
    key_provider = MockKeyProvider()
    nominal_key = ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    
    metric = graph.MetricNode(metric_spec, key_provider, nominal_key)
    symbol.add_child(metric)
    
    # Add analyzer
    analyzer = graph.AnalyzerNode(MockOp("test_analyzer"))
    metric.add_child(analyzer)
    
    # Get tree representation
    tree = root.inspect()
    
    # Verify tree structure
    assert tree.label == "Suite: Test Suite"
    assert len(tree.children) == 1  # One check node
    
    check_tree = tree.children[0]
    assert "Check One" in str(check_tree.label)
    
    # Should have assertion and symbol as children
    assert len(check_tree.children) == 2
    
    # Find the symbol child (assertion might be filtered out if no validator)
    symbol_tree = None
    for child in check_tree.children:
        if "symbol1" in str(child.label):
            symbol_tree = child
            break
    
    assert symbol_tree is not None, "Symbol node not found in tree"
    assert len(symbol_tree.children) == 1  # One metric
    
    metric_tree = symbol_tree.children[0]
    assert "test_metric" in str(metric_tree.label)
    assert len(metric_tree.children) == 1  # One analyzer
    
    analyzer_tree = metric_tree.children[0]
    assert "test_analyzer" in str(analyzer_tree.label)


# Tests from test_api.py (graph-specific tests)
def test_inspect_no_run() -> None:
    from dqx.api import VerificationSuite, check
    from dqx.common import Context, MetricProvider
    from dqx.orm.repositories import InMemoryMetricDB
    from dqx import specs
    
    @check(datasets=["abc"])
    def simple_checks(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.null_count("delivered")).is_leq(100)
        ctx.assert_that(mp.minimum("quantity")).is_leq(2.5)
        ctx.assert_that(mp.average("price")).is_geq(10.0)
        ctx.assert_that(mp.ext.day_over_day(specs.Average("tax"))).is_geq(0.5)

    @check(label="Delivered null percentage", datasets=["ds1"])
    def null_percentage(mp: MetricProvider, ctx: Context) -> None:
        null_count = mp.null_count("delivered", datasets=["ds1"])
        nr = mp.num_rows()
        ctx.assert_that(null_count / nr).on(label="null percentage is less than 40%").is_leq(0.4)

    @check(label="Manual day-over-day check", datasets=["ds1"])
    def manual_day_over_day(mp: MetricProvider, ctx: Context) -> None:
        tax_avg = mp.average("tax")
        tax_avg_lag = mp.average("tax", key=ctx.key.lag(1))
        ctx.assert_that(tax_avg / tax_avg_lag).on().is_eq(1.0, tol=0.01)

    @check(label="Rate of change", datasets=["ds2"])
    def rate_of_change(mp: MetricProvider, ctx: Context) -> None:
        tax_avg = mp.ext.day_over_day(specs.Maximum("tax"))
        rate = sp.Abs(tax_avg - 1.0)
        ctx.assert_that(rate).on(label="Maximum tax rate change is less than 20%").is_leq(0.2)

    @check(datasets=["ds1"])
    def sketch_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.approx_cardinality("address", datasets=["ds2"])).is_geq(100)
    
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})
    checks = [simple_checks, manual_day_over_day, rate_of_change, null_percentage, sketch_check]

    # Run once for yesterday
    suite = VerificationSuite(checks, db, name="Simple test suite")
    ctx = suite.collect(key)
    ctx._graph.propagate(["ds1", "ds2"])
    tree = ctx._graph.inspect()
    Console().print(tree)


def test_assertion_node_cannot_have_children() -> None:
    """Test that AssertionNode raises error when trying to add children."""
    # Create an assertion node
    assertion = AssertionNode(actual=sp.Symbol("x"))
    
    # Create a symbol node
    symbol = SymbolNode(
        name="test_symbol",
        symbol=sp.Symbol("x"),
        fn=lambda k: Success(10.0),
        datasets=["ds1"]
    )
    
    # Try to add symbol as child - should raise RuntimeError
    with pytest.raises(RuntimeError, match="AssertionNode cannot have children"):
        assertion.add_child(symbol)


def test_assertion_node_starts_with_empty_children() -> None:
    """Test that AssertionNode initializes with empty children list."""
    assertion = AssertionNode(actual=sp.Symbol("x"))
    assert assertion.children == []
    assert len(assertion.children) == 0


def test_check_node_can_have_assertion_and_symbol_children() -> None:
    """Test that CheckNode can have both AssertionNode and SymbolNode children."""
    # Create a check node
    check = CheckNode(name="test_check")
    
    # Create assertion and symbol nodes
    assertion = AssertionNode(actual=sp.Symbol("x"))
    symbol = SymbolNode(
        name="test_symbol",
        symbol=sp.Symbol("x"),
        fn=lambda k: Success(10.0),
        datasets=["ds1"]
    )
    
    # Add both as children - should work fine
    check.add_child(assertion)
    check.add_child(symbol)
    
    assert len(check.children) == 2
    assert assertion in check.children
    assert symbol in check.children


def test_cleanup_assertion_children() -> None:
    """Test that cleanup method removes any legacy children from assertion nodes."""
    # Create a root node and structure
    root = RootNode(name="Test Suite")
    check = CheckNode(name="test_check")
    root.add_child(check)
    
    # Create an assertion node
    assertion = AssertionNode(actual=sp.Symbol("x"), root=root)
    check.add_child(assertion)
    
    # Manually add some children to assertion (simulating legacy behavior)
    # We bypass the add_child method to simulate old code
    assertion.children.append("legacy_child_1")
    assertion.children.append("legacy_child_2")
    
    # Verify children were added
    assert len(assertion.children) == 2
    
    # Run propagate which should clean up
    root.propagate(["ds1"])
    
    # Verify children were removed
    assert len(assertion.children) == 0
    assert assertion.children == []


def test_assertion_propagate_does_nothing() -> None:
    """Test that AssertionNode.propagate() does nothing (no children to propagate to)."""
    assertion = AssertionNode(actual=sp.Symbol("x"))
    
    # This should not raise any errors even though there are no children
    assertion.propagate()
    
    # Still no children
    assert assertion.children == []


def test_graph_structure_without_assertion_children() -> None:
    """Test that the graph structure works correctly without assertion children."""
    # Create a complete graph structure
    root = RootNode(name="Test Suite")
    check = CheckNode(name="test_check")
    root.add_child(check)
    
    # Create assertion
    assertion = AssertionNode(
        actual=sp.Symbol("x_1") + sp.Symbol("x_2"),
        root=root
    )
    check.add_child(assertion)
    
    # Create symbols as children of check (not assertion)
    symbol1 = SymbolNode(
        name="metric1",
        symbol=sp.Symbol("x_1"),
        fn=lambda k: Success(10.0),
        datasets=[]
    )
    symbol2 = SymbolNode(
        name="metric2", 
        symbol=sp.Symbol("x_2"),
        fn=lambda k: Success(20.0),
        datasets=[]
    )
    check.add_child(symbol1)
    check.add_child(symbol2)
    
    # Verify structure
    assert len(root.children) == 1
    assert len(check.children) == 3  # 1 assertion + 2 symbols
    assert len(assertion.children) == 0  # No children
    
    # Verify we can still find all nodes
    assertions = list(root.assertions())
    symbols = list(root.symbols())
    
    assert len(assertions) == 1
    assert len(symbols) == 2
    assert assertions[0] == assertion
    assert symbol1 in symbols
    assert symbol2 in symbols
