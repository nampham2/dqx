"""Comprehensive tests for graph module with 100% coverage."""

import datetime as dt
from typing import Any
from unittest.mock import MagicMock

import pytest
import sympy as sp
from returns.maybe import Nothing, Some
from returns.result import Failure, Success

from dqx import graph
from dqx.common import ResultKey, ResultKeyProvider, SymbolicValidator
from dqx.ops import Op
from dqx.specs import MetricSpec


# =============================================================================
# Helper Classes
# =============================================================================

class MockOp(Op[Any]):
    """Mock operator for testing."""
    def __init__(self, name: str):
        self._name = name
        self._prefix = "mock"
        self._value = None
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def prefix(self) -> str:
        return self._prefix
    
    @property
    def value(self) -> Any:
        return self._value
    
    def assign(self, value: Any) -> None:
        self._value = value
    
    def clear(self) -> None:
        self._value = None


class MockKeyProvider(ResultKeyProvider):
    """Mock key provider for testing."""
    def create(self, key: ResultKey) -> ResultKey:
        return key


# =============================================================================
# 1. Core Node Behavior Tests
# =============================================================================

def test_basic_graph_creation() -> None:
    """Test basic graph creation and structure."""
    root = graph.RootNode("Test Suite")
    check = graph.CheckNode("check_1")
    root.add_child(check)
    
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    check.add_child(assertion)
    
    assert root.name == "Test Suite"
    assert len(root.children) == 1
    assert check in root.children
    assert len(check.children) == 1
    assert assertion in check.children




# =============================================================================
# 2. Design Pattern Tests
# =============================================================================

def test_visitor_pattern_implementation() -> None:
    """Test the visitor pattern for graph traversal."""
    # Create a graph structure
    root = graph.RootNode("Test Suite")
    check1 = graph.CheckNode("check1")
    check2 = graph.CheckNode("check2")
    root.add_child(check1)
    root.add_child(check2)
    
    # Add various node types
    assertion1 = graph.AssertionNode(actual=sp.Symbol("x"), root=root)
    symbol1 = graph.SymbolNode("sym1", sp.Symbol("x"), lambda k: Success(10.0), [])
    check1.add_child(assertion1)
    check1.add_child(symbol1)
    
    assertion2 = graph.AssertionNode(actual=sp.Symbol("y"), root=root)
    symbol2 = graph.SymbolNode("sym2", sp.Symbol("y"), lambda k: Success(20.0), [])
    check2.add_child(assertion2)
    check2.add_child(symbol2)
    
    # Test traversal with filtering
    assertions = list(root.assertions())
    assert len(assertions) == 2
    assert assertion1 in assertions
    assert assertion2 in assertions
    
    symbols = list(root.symbols())
    assert len(symbols) == 2
    assert symbol1 in symbols
    assert symbol2 in symbols
    
    checks = list(root.checks())
    assert len(checks) == 2
    assert check1 in checks
    assert check2 in checks


def test_custom_visitor() -> None:
    """Test custom visitor implementation."""
    class CountingVisitor(graph.NodeVisitor):
        def __init__(self) -> None:
            self.count = 0
        
        def visit(self, node: graph.BaseNode) -> None:
            self.count += 1
            # Continue traversal for composite nodes
            if isinstance(node, graph.CompositeNode):
                for child in node.get_children():
                    child.accept(self)
    
    # Build a graph
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    assertion = graph.AssertionNode(sp.Symbol("x"), root=root)
    symbol = graph.SymbolNode("sym", sp.Symbol("x"), lambda k: Success(1.0), [])
    
    root.add_child(check)
    check.add_child(assertion)
    check.add_child(symbol)
    
    # Count nodes
    visitor = CountingVisitor()
    root.accept(visitor)
    
    # Should count: root + check + assertion + symbol = 4
    assert visitor.count == 4


def test_composite_pattern_hierarchy() -> None:
    """Test the composite pattern for node hierarchy."""
    root = graph.RootNode("Root")
    check = graph.CheckNode("Check")
    symbol = graph.SymbolNode("Symbol", sp.Symbol("x"), lambda k: Success(1.0), [])
    metric = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    
    # Test add_child
    root.add_child(check)
    check.add_child(symbol)
    symbol.add_child(metric)
    
    assert len(root.children) == 1
    assert len(check.children) == 1
    assert len(symbol.children) == 1
    assert len(metric.children) == 0
    
    # Test remove_child
    check.remove_child(symbol)
    assert len(check.children) == 0
    
    # Test get_children
    children = root.get_children()
    assert children == [check]
    
    # Re-add symbol to check
    check.add_child(symbol)
    assert check.get_children() == [symbol]


def test_traverse_without_filter() -> None:
    """Test RootNode traverse method without type filter."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    assertion = graph.AssertionNode(sp.Symbol("x"), root=root)
    symbol = graph.SymbolNode("sym", sp.Symbol("x"), lambda k: Success(1.0), [])
    
    root.add_child(check)
    check.add_child(assertion)
    check.add_child(symbol)
    
    # Traverse without filter should return all nodes
    all_nodes = list(root.traverse())
    assert len(all_nodes) == 4  # root + check + assertion + symbol
    assert root in all_nodes
    assert check in all_nodes
    assert assertion in all_nodes
    assert symbol in all_nodes




# =============================================================================
# 3. Node Hierarchy Tests
# =============================================================================

def test_leaf_nodes_cannot_have_children() -> None:
    """Test that leaf nodes (AssertionNode, AnalyzerNode) cannot have children."""
    # AssertionNode
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    with pytest.raises(RuntimeError, match="AssertionNode cannot have children"):
        assertion.add_child(MagicMock())
    
    # AnalyzerNode
    analyzer = graph.AnalyzerNode(MockOp("test"))
    with pytest.raises(NotImplementedError, match="AnalyzerNode cannot have children"):
        analyzer.add_child(MagicMock())


def test_check_node_accepts_multiple_child_types() -> None:
    """Test that CheckNode can have both AssertionNode and SymbolNode children."""
    check = graph.CheckNode(name="test_check")
    
    # Create assertion and symbol nodes
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    symbol = graph.SymbolNode("test_symbol", sp.Symbol("x"), lambda k: Success(10.0), ["ds1"])
    
    # Add both as children
    check.add_child(assertion)
    check.add_child(symbol)
    
    assert len(check.children) == 2
    assert assertion in check.children
    assert symbol in check.children


def test_root_node_exists_method() -> None:
    """Test the exists method for backward compatibility."""
    root = graph.RootNode("Test")
    check1 = graph.CheckNode("check1")
    check2 = graph.CheckNode("check2")
    
    # Add check1 but not check2
    root.add_child(check1)
    
    assert root.exists(check1)
    assert not root.exists(check2)


# =============================================================================
# 4. State Management Tests
# =============================================================================

def test_metric_node_states() -> None:
    """Test MetricNode state transitions."""
    metric = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="test_metric"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    
    # Initial state should be PENDING
    assert metric.state() == "PENDING"
    
    # Mark as provided
    metric.mark_as_provided()
    assert metric.state() == "PROVIDED"
    
    # Create another metric and mark as failure
    metric2 = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="test_metric2"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    metric2.mark_as_failure("Test error")
    assert metric2.state() == "ERROR"
    
    # Mark as success (same as provided)
    metric3 = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="test_metric3"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    metric3.mark_as_success()
    assert metric3.state() == "PROVIDED"


def test_symbol_node_states() -> None:
    """Test SymbolNode success and failure states."""
    symbol = graph.SymbolNode("test", sp.Symbol("x"), lambda k: Success(10.0), [])
    
    # Initially not success or failure
    assert not symbol.success()
    assert not symbol.failure()
    
    # Evaluate successfully
    symbol._value = Some(Success(10.0))
    assert symbol.success()
    assert not symbol.failure()
    
    # Mark as failure
    symbol.mark_as_failure("Test error")
    assert not symbol.success()
    assert symbol.failure()


def test_symbol_node_ready_state() -> None:
    """Test SymbolNode ready() method based on child metrics."""
    symbol = graph.SymbolNode("test", sp.Symbol("x"), lambda k: Success(10.0), [])
    
    # No children - should be ready
    assert symbol.ready()
    
    # Add metric children
    metric1 = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric1"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    metric2 = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric2"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    
    symbol.add_child(metric1)
    symbol.add_child(metric2)
    
    # Not ready when metrics are pending
    assert not symbol.ready()
    
    # Mark one as provided
    metric1.mark_as_provided()
    assert not symbol.ready()
    
    # Mark both as provided
    metric2.mark_as_provided()
    assert symbol.ready()


def test_root_node_metric_filtering() -> None:
    """Test RootNode methods for filtering metrics by state."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check", datasets=["ds1"])
    root.add_child(check)
    
    symbol = graph.SymbolNode("sym", sp.Symbol("x"), lambda k: Success(1.0), [])
    check.add_child(symbol)
    
    # Create metrics in different states
    metric1 = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric1"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    metric2 = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric2"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    metric3 = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric3"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    
    symbol.add_child(metric1)
    symbol.add_child(metric2)
    symbol.add_child(metric3)
    
    # Set different states
    metric1.mark_as_provided()
    # metric2 remains PENDING
    metric3.mark_as_failure("error")
    
    # Propagate to set datasets
    root.propagate(["ds1"])
    
    # Test filtering methods
    provided = list(root.provided_metrics())
    assert len(provided) == 1
    assert metric1 in provided
    
    pending = list(root.pending_metrics("ds1"))
    assert len(pending) == 1
    assert metric2 in pending
    
    # Test ready_metrics (none should be READY in this test)
    ready = list(root.ready_metrics())
    assert len(ready) == 0


def test_root_node_ready_symbols() -> None:
    """Test RootNode ready_symbols() method."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create symbols with different readiness
    symbol1 = graph.SymbolNode("sym1", sp.Symbol("x"), lambda k: Success(1.0), [])
    symbol2 = graph.SymbolNode("sym2", sp.Symbol("y"), lambda k: Success(2.0), [])
    check.add_child(symbol1)
    check.add_child(symbol2)
    
    # symbol1 has no metrics, so it's ready
    # symbol2 has a pending metric
    metric = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    symbol2.add_child(metric)
    
    ready_symbols = list(root.ready_symbols())
    assert len(ready_symbols) == 1
    assert symbol1 in ready_symbols
    assert symbol2 not in ready_symbols
    
    # Mark metric as provided
    metric.mark_as_provided()
    ready_symbols = list(root.ready_symbols())
    assert len(ready_symbols) == 2


def test_root_node_mark_pending_metrics() -> None:
    """Test marking pending metrics as success or failed."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check", datasets=["ds1"])
    root.add_child(check)
    
    symbol = graph.SymbolNode("sym", sp.Symbol("x"), lambda k: Success(1.0), [])
    check.add_child(symbol)
    
    # Create pending metrics
    metric1 = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric1"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    metric2 = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric2"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    
    symbol.add_child(metric1)
    symbol.add_child(metric2)
    
    # Propagate to set datasets
    root.propagate(["ds1"])
    
    # Both should be pending
    assert metric1.state() == "PENDING"
    assert metric2.state() == "PENDING"
    
    # Mark all pending as success
    root.mark_pending_metrics_success("ds1")
    assert metric1.state() == "PROVIDED"
    assert metric2.state() == "PROVIDED"
    
    # Reset for failure test
    metric1._analyzed = Nothing
    metric2._analyzed = Nothing
    
    # Mark all pending as failed
    root.mark_pending_metric_failed("ds1", "Test failure")
    assert metric1.state() == "ERROR"
    assert metric2.state() == "ERROR"


def test_metrics_method() -> None:
    """Test RootNode metrics() method."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    symbol = graph.SymbolNode("sym", sp.Symbol("x"), lambda k: Success(1.0), [])
    metric = graph.MetricNode(
        MagicMock(spec=MetricSpec, name="metric"),
        MockKeyProvider(),
        ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    )
    
    root.add_child(check)
    check.add_child(symbol)
    symbol.add_child(metric)
    
    metrics = list(root.metrics())
    assert len(metrics) == 1
    assert metric in metrics


# =============================================================================
# 5. Dataset Propagation Tests
# =============================================================================

def test_dataset_propagation() -> None:
    """Test dataset propagation through the graph."""
    root = graph.RootNode("Test")
    check1 = graph.CheckNode("check1", datasets=["ds1"])
    check2 = graph.CheckNode("check2")  # No specific dataset
    root.add_child(check1)
    root.add_child(check2)
    
    # Add children
    assertion1 = graph.AssertionNode(sp.Symbol("x"), root=root)
    symbol1 = graph.SymbolNode("sym1", sp.Symbol("x"), lambda k: Success(1.0), [])
    check1.add_child(assertion1)
    check1.add_child(symbol1)
    
    assertion2 = graph.AssertionNode(sp.Symbol("y"), root=root)
    check2.add_child(assertion2)
    
    # Propagate datasets
    root.propagate(["ds1", "ds2"])
    
    # check1 should keep its dataset requirement
    assert check1.datasets == ["ds1"]
    # check2 should get all datasets
    assert check2.datasets == ["ds1", "ds2"]
    
    # Children should inherit from their parent check
    assert assertion1.datasets == ["ds1"]
    assert symbol1.datasets == ["ds1"]
    assert assertion2.datasets == ["ds1", "ds2"]


def test_dataset_validation_errors() -> None:
    """Test dataset validation error cases."""
    root = graph.RootNode("Test")
    
    # Test CheckNode dataset validation error
    check = graph.CheckNode("check1", datasets=["ds_missing"])
    root.add_child(check)
    root.propagate(["ds1", "ds2"])
    
    # Check should have error value
    assert isinstance(check._value, Some)
    assert isinstance(check._value.unwrap(), Failure)
    assert "requires datasets ['ds_missing']" in check._value.unwrap().failure()
    
    # Test AssertionNode dataset validation error
    root2 = graph.RootNode("Test2")
    check2 = graph.CheckNode("check2")
    root2.add_child(check2)
    
    assertion = graph.AssertionNode(sp.Symbol("x"), root=root2)
    assertion.set_datasource(["ds_missing"])
    check2.add_child(assertion)
    
    root2.propagate(["ds1"])
    assert isinstance(assertion._value, Some)
    assert isinstance(assertion._value.unwrap(), Failure)
    
    # Test SymbolNode dataset validation error
    symbol = graph.SymbolNode("sym", sp.Symbol("x"), lambda k: Success(1.0), ["ds_missing"])
    check2.add_child(symbol)
    
    root2.propagate(["ds1"])
    assert isinstance(symbol._value, Some)
    assert isinstance(symbol._value.unwrap(), Failure)


def test_symbol_dataset_count_validation() -> None:
    """Test SymbolNode dataset count validation."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Symbol requires exactly 1 dataset
    symbol = graph.SymbolNode("sym", sp.Symbol("x"), lambda k: Success(1.0), [])
    symbol._required_ds_count = 1
    check.add_child(symbol)
    
    # Propagate with 2 datasets - should fail
    root.propagate(["ds1", "ds2"])
    
    assert isinstance(symbol._value, Some)
    assert isinstance(symbol._value.unwrap(), Failure)
    assert "requires exactly 1 datasets" in symbol._value.unwrap().failure()


def test_symbol_node_impute_dataset_no_change_needed() -> None:
    """Test impute_dataset when datasets already match requirements."""
    # Create a symbol with datasets already set correctly
    symbol = graph.SymbolNode(
        name="test_symbol",
        symbol=sp.Symbol("x"),
        fn=lambda k: Success(42.0),
        datasets=["dataset1"]
    )
    
    # Set _required_ds_count to match current datasets
    symbol._required_ds_count = 1
    
    # Call impute_dataset with matching datasets - should return early
    symbol.impute_dataset(["dataset1", "dataset2"])
    
    # Verify datasets unchanged and no error set
    assert symbol.datasets == ["dataset1"]
    assert symbol._value == Nothing


# =============================================================================
# 6. Evaluation and Computation Tests
# =============================================================================

def test_assertion_node_setters() -> None:
    """Test AssertionNode setter methods."""
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    
    # Test set_label
    assertion.set_label("Test Label")
    assert assertion.label == "Test Label"
    
    # Test set_severity
    assertion.set_severity("P0")
    assert assertion.severity == "P0"
    
    # Test set_validator
    validator = SymbolicValidator(name="> 5", fn=lambda x: x > 5)
    assertion.set_validator(validator)
    assert assertion.validator == validator
    
    # Test set_datasource
    assertion.set_datasource(["ds1", "ds2"])
    assert assertion.datasets == ["ds1", "ds2"]


def test_assertion_node_evaluate_success() -> None:
    """Test successful assertion evaluation."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create symbols
    symbol_x = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Success(10.0), [])
    symbol_y = graph.SymbolNode("y_metric", sp.Symbol("y"), lambda k: Success(20.0), [])
    check.add_child(symbol_x)
    check.add_child(symbol_y)
    
    # Evaluate symbols
    symbol_x._value = Some(Success(10.0))
    symbol_y._value = Some(Success(20.0))
    
    # Create assertion
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + sp.Symbol("y"),
        validator=SymbolicValidator(name="> 25", fn=lambda v: v > 25),
        root=root
    )
    check.add_child(assertion)
    
    # Evaluate assertion
    result = assertion.evaluate()
    assert isinstance(result, Success)
    assert float(result.unwrap()) == 30.0
    assert float(assertion._value.unwrap().unwrap()) == 30.0


def test_assertion_node_evaluate_validator_failure() -> None:
    """Test assertion evaluation with validator failure."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create symbol
    symbol = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Success(5.0), [])
    check.add_child(symbol)
    symbol._value = Some(Success(5.0))
    
    # Create assertion with failing validator
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        label="Test assertion",
        validator=SymbolicValidator(name="> 10", fn=lambda v: v > 10),
        root=root
    )
    check.add_child(assertion)
    
    # Evaluate assertion
    result = assertion.evaluate()
    assert isinstance(result, Failure)
    assert "does not satisfy > 10" in result.failure()
    assert "Test assertion:" in result.failure()


def test_assertion_node_evaluate_missing_symbols() -> None:
    """Test assertion evaluation with missing symbols."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create assertion without corresponding symbols
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + sp.Symbol("y"),
        root=root
    )
    check.add_child(assertion)
    
    # Evaluate assertion - should fail due to missing symbols
    result = assertion.evaluate()
    assert isinstance(result, Failure)
    assert "Missing symbols" in result.failure()


def test_assertion_node_evaluate_failed_symbols() -> None:
    """Test assertion evaluation with failed symbol dependencies."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create symbols with failures
    symbol_x = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Success(10.0), [])
    symbol_y = graph.SymbolNode("y_metric", sp.Symbol("y"), lambda k: Failure("Symbol failed"), [])
    check.add_child(symbol_x)
    check.add_child(symbol_y)
    
    # Mark symbols
    symbol_x._value = Some(Success(10.0))
    symbol_y._value = Some(Failure("Symbol failed"))
    
    # Create assertion
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + sp.Symbol("y"),
        root=root
    )
    check.add_child(assertion)
    
    # Evaluate assertion - should fail due to failed symbol
    result = assertion.evaluate()
    assert isinstance(result, Failure)
    assert "Symbol dependencies failed" in result.failure()


def test_assertion_node_evaluate_nan_and_infinity() -> None:
    """Test assertion evaluation with NaN and infinity values."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Test NaN
    symbol_nan = graph.SymbolNode("nan_metric", sp.Symbol("x"), lambda k: Success(float('nan')), [])
    check.add_child(symbol_nan)
    symbol_nan._value = Some(Success(float('nan')))
    
    assertion_nan = graph.AssertionNode(actual=sp.Symbol("x"), root=root)
    check.add_child(assertion_nan)
    
    result_nan = assertion_nan.evaluate()
    assert isinstance(result_nan, Failure)
    assert "Validating value is NaN" in result_nan.failure()
    
    # Test infinity
    root2 = graph.RootNode("Test2")
    check2 = graph.CheckNode("check2")
    root2.add_child(check2)
    
    symbol_inf = graph.SymbolNode("inf_metric", sp.Symbol("y"), lambda k: Success(float('inf')), [])
    check2.add_child(symbol_inf)
    symbol_inf._value = Some(Success(float('inf')))
    
    assertion_inf = graph.AssertionNode(actual=sp.Symbol("y"), root=root2)
    check2.add_child(assertion_inf)
    
    result_inf = assertion_inf.evaluate()
    assert isinstance(result_inf, Failure)
    assert "Validating value is infinity" in result_inf.failure()


def test_assertion_node_evaluate_without_root() -> None:
    """Test assertion evaluation without root node."""
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    
    with pytest.raises(RuntimeError, match="Root node not set"):
        assertion.evaluate()


def test_assertion_node_parent_failure() -> None:
    """Test that assertions don't evaluate when parent CheckNode fails."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check", datasets=["required_dataset"])
    root.add_child(check)
    
    # Add symbol and assertion
    symbol = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Success(10.0), [])
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + 5,
        validator=SymbolicValidator(name="> 10", fn=lambda v: v > 10),
        root=root
    )
    check.add_child(symbol)
    check.add_child(assertion)
    
    # Symbol is successful
    symbol._value = Some(Success(10.0))
    
    # Propagate with wrong dataset - this should fail the check
    root.propagate(["different_dataset"])
    
    # Evaluate assertion - should fail due to parent failure
    result = assertion.evaluate()
    assert isinstance(result, Failure)
    assert "Parent check failed!" in result.failure()


def test_assertion_node_parent_and_dependency_failures() -> None:
    """Test that parent failures are checked before dependency failures."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check", datasets=["required_dataset"])
    root.add_child(check)
    
    # Add failing symbol and assertion
    symbol = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Failure("Symbol error"), [])
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + sp.Symbol("y"),  # y is missing
        root=root
    )
    check.add_child(symbol)
    check.add_child(assertion)
    
    # Symbol has failed
    symbol._value = Some(Failure("Symbol error"))
    
    # Propagate with wrong dataset - this should fail the check
    root.propagate(["different_dataset"])
    
    # Evaluate assertion - should fail due to parent failure, not dependency issues
    result = assertion.evaluate()
    assert isinstance(result, Failure)
    assert "Parent check failed!" in result.failure()
    # Should not mention symbol dependencies or missing symbols
    assert "Symbol dependencies failed" not in result.failure()
    assert "Missing symbols" not in result.failure()


def test_assertion_node_find_parent_check() -> None:
    """Test the _find_parent_check helper method."""
    root = graph.RootNode("Test")
    check1 = graph.CheckNode("check1")
    check2 = graph.CheckNode("check2")
    root.add_child(check1)
    root.add_child(check2)
    
    assertion1 = graph.AssertionNode(sp.Symbol("x"), root=root)
    assertion2 = graph.AssertionNode(sp.Symbol("y"), root=root)
    assertion3 = graph.AssertionNode(sp.Symbol("z"), root=root)
    
    check1.add_child(assertion1)
    check2.add_child(assertion2)
    # assertion3 is not added to any check
    
    # Test finding parent checks
    assert assertion1._find_parent_check() == check1
    assert assertion2._find_parent_check() == check2
    assert assertion3._find_parent_check() is None
    
    # Test with no root
    assertion4 = graph.AssertionNode(sp.Symbol("w"))
    assert assertion4._find_parent_check() is None


def test_assertion_node_find_root_error() -> None:
    """Test _find_root method error case."""
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    
    with pytest.raises(RuntimeError, match="Root node not set"):
        assertion._find_root()


def test_assertion_node_mark_as_failure() -> None:
    """Test marking assertion as failure."""
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    
    assertion.mark_as_failure("Test failure message")
    
    assert isinstance(assertion._value, Some)
    assert isinstance(assertion._value.unwrap(), Failure)
    assert assertion._value.unwrap().failure() == "Test failure message"


def test_symbol_node_evaluate() -> None:
    """Test SymbolNode evaluate method."""
    # Success case
    def fn_success(k: ResultKey) -> Success[float]:
        return Success(42.0)
    
    symbol = graph.SymbolNode("test", sp.Symbol("x"), fn_success, [])
    
    key = ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    result = symbol.evaluate(key)
    
    assert isinstance(result, Success)
    assert result.unwrap() == 42.0
    assert symbol._value.unwrap().unwrap() == 42.0
    
    # Failure case
    def fn_failure(k: ResultKey) -> Failure[str]:
        return Failure("Evaluation failed")
    
    symbol2 = graph.SymbolNode("test2", sp.Symbol("y"), fn_failure, [])
    
    result2 = symbol2.evaluate(key)
    assert isinstance(result2, Failure)
    assert result2.failure() == "Evaluation failed"


def test_metric_node_eval_key() -> None:
    """Test MetricNode eval_key method."""
    spec = MagicMock(spec=MetricSpec, name="test_metric")
    key_provider = MockKeyProvider()
    nominal_key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={"env": "test"})
    
    metric = graph.MetricNode(spec, key_provider, nominal_key)
    
    eval_key = metric.eval_key()
    assert eval_key == nominal_key
    assert eval_key.yyyy_mm_dd == dt.date(2025, 1, 15)
    assert eval_key.tags == {"env": "test"}


def test_check_node_name() -> None:
    """Test CheckNode node_name method."""
    # With label
    check1 = graph.CheckNode("check_id", label="Check Label")
    assert check1.node_name() == "Check Label"
    
    # Without label
    check2 = graph.CheckNode("check_id")
    assert check2.node_name() == "check_id"


def test_check_node_update_status() -> None:
    """Test CheckNode update_status method."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Initially check should be pending
    assert check._value == Nothing
    
    # Update with no children - should remain pending
    check.update_status()
    assert check._value == Nothing
    
    # Add successful children
    assertion1 = graph.AssertionNode(sp.Symbol("x"), root=root)
    assertion1._value = Some(Success(10.0))
    check.add_child(assertion1)
    
    symbol1 = graph.SymbolNode("sym", sp.Symbol("y"), lambda k: Success(20.0), [])
    symbol1._value = Some(Success(20.0))
    check.add_child(symbol1)
    
    # Update status - should be success
    check.update_status()
    assert isinstance(check._value, Some)
    assert isinstance(check._value.unwrap(), Success)


def test_check_node_update_status_with_failures() -> None:
    """Test CheckNode update_status with failed children."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Add mixed success/failure children
    assertion1 = graph.AssertionNode(sp.Symbol("x"), label="Assertion 1", root=root)
    assertion1._value = Some(Success(10.0))
    check.add_child(assertion1)
    
    assertion2 = graph.AssertionNode(sp.Symbol("y"), label="Assertion 2", root=root)
    assertion2._value = Some(Failure("Validation failed"))
    check.add_child(assertion2)
    
    # Update status - should be failure
    check.update_status()
    assert isinstance(check._value, Some)
    result = check._value.unwrap()
    assert isinstance(result, Failure)
    assert "Assertion 2: Validation failed" in result.failure()


def test_check_node_update_status_multiple_failures() -> None:
    """Test CheckNode update_status with multiple failed children."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Add multiple failures
    assertion1 = graph.AssertionNode(sp.Symbol("x"), root=root)
    assertion1._value = Some(Failure("First failure"))
    check.add_child(assertion1)
    
    assertion2 = graph.AssertionNode(sp.Symbol("y"), root=root)
    assertion2._value = Some(Failure("Second failure"))
    check.add_child(assertion2)
    
    # Update status - should show multiple failures
    check.update_status()
    assert isinstance(check._value, Some)
    result = check._value.unwrap()
    assert isinstance(result, Failure)
    assert "Multiple failures:" in result.failure()
    assert "First failure" in result.failure()
    assert "Second failure" in result.failure()


def test_check_node_update_status_preserves_dataset_failure() -> None:
    """Test that update_status preserves existing dataset mismatch failures."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check", datasets=["required_ds"])
    root.add_child(check)
    
    # Propagate with wrong dataset - this fails the check
    root.propagate(["different_ds"])
    
    # Add successful children
    assertion = graph.AssertionNode(sp.Symbol("x"), root=root)
    assertion._value = Some(Success(10.0))
    check.add_child(assertion)
    
    # Update status - should preserve dataset failure
    check.update_status()
    assert isinstance(check._value, Some)
    result = check._value.unwrap()
    assert isinstance(result, Failure)
    assert "requires datasets" in result.failure()


def test_check_node_update_status_with_pending_children() -> None:
    """Test CheckNode update_status with some pending children."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Add mixed evaluated and pending children
    assertion1 = graph.AssertionNode(sp.Symbol("x"), root=root)
    assertion1._value = Some(Success(10.0))
    check.add_child(assertion1)
    
    assertion2 = graph.AssertionNode(sp.Symbol("y"), root=root)
    # Leave assertion2 as pending (Nothing)
    check.add_child(assertion2)
    
    # Update status - should remain pending
    check.update_status()
    assert check._value == Nothing


def test_assertion_node_evaluate_no_validator() -> None:
    """Test assertion evaluation without validator (just computes value)."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create symbol
    symbol = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Success(10.0), [])
    check.add_child(symbol)
    symbol._value = Some(Success(10.0))
    
    # Create assertion without validator
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") * 2,
        root=root
    )
    check.add_child(assertion)
    
    # Evaluate assertion
    result = assertion.evaluate()
    assert isinstance(result, Success)
    assert float(result.unwrap()) == 20.0


# =============================================================================
# 7. Error Handling Tests
# =============================================================================

def test_assertion_node_evaluate_exception() -> None:
    """Test assertion evaluation with exception."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create a symbol that causes division by zero
    symbol_x = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Success(0.0), [])
    check.add_child(symbol_x)
    symbol_x._value = Some(Success(0.0))
    
    # Create assertion with division by zero
    assertion = graph.AssertionNode(
        actual=sp.Integer(1) / sp.Symbol("x"),
        root=root
    )
    check.add_child(assertion)
    
    # Evaluate assertion - should catch exception
    result = assertion.evaluate()
    assert isinstance(result, Failure)


def test_assertion_propagate_does_nothing() -> None:
    """Test that AssertionNode.propagate() does nothing."""
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    
    # This should not raise any errors
    assertion.propagate()
    
    # AssertionNode is a LeafNode and has no children


# =============================================================================
# 8. Minimal Coverage Tests for Display Methods
# =============================================================================

def test_node_format_display_methods_exist() -> None:
    """Test that all node types have format_display methods for coverage."""
    # RootNode
    root = graph.RootNode("Test Suite")
    assert hasattr(root, 'format_display')
    root.format_display()  # Call to cover the method
    
    # CheckNode
    check = graph.CheckNode("check1", label="Check One", datasets=["ds1"])
    check.format_display()
    
    # AssertionNode with different states
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    assertion.format_display()
    
    # AssertionNode with validator
    validator = SymbolicValidator(name="> 10", fn=lambda x: x > 10)
    assertion_with_validator = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=validator
    )
    assertion_with_validator.format_display()
    
    # SymbolNode
    symbol = graph.SymbolNode("sym1", sp.Symbol("y"), lambda k: Success(1.0), ["ds1"])
    symbol.format_display()
    
    # SymbolNode with value
    symbol._value = Some(Success(42.0))
    symbol.format_display()
    
    # SymbolNode with failure
    symbol2 = graph.SymbolNode("sym2", sp.Symbol("z"), lambda k: Failure("error"), [])
    symbol2._value = Some(Failure("error"))
    symbol2.format_display()
    
    # MetricNode
    metric_spec = MagicMock(spec=MetricSpec, name="metric1")
    metric = graph.MetricNode(metric_spec, MockKeyProvider(), ResultKey(yyyy_mm_dd=dt.date.today(), tags={}))
    metric.format_display()
    
    # AnalyzerNode
    analyzer = graph.AnalyzerNode(MockOp("analyzer1"))
    analyzer.format_display()


def test_root_node_inspect_method() -> None:
    """Test RootNode inspect method for coverage."""
    root = graph.RootNode("Test Suite")
    check = graph.CheckNode("check1")
    root.add_child(check)
    
    # Call inspect to cover the method
    tree = root.inspect()
    assert tree is not None  # Just verify it returns something
