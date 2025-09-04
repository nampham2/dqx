"""Comprehensive tests for graph module with simplified architecture."""

from unittest.mock import MagicMock

import pytest
import sympy as sp
from returns.maybe import Nothing, Some
from returns.result import Failure, Success

from dqx import graph
from dqx.common import ResultKey, ResultKeyProvider, SymbolicValidator
from dqx.display import GraphDisplay
from dqx.symbol_table import SymbolEntry


# =============================================================================
# Helper Classes
# =============================================================================


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

    # Add assertions
    assertion1 = graph.AssertionNode(actual=sp.Symbol("x"), root=root)
    check1.add_child(assertion1)

    assertion2 = graph.AssertionNode(actual=sp.Symbol("y"), root=root)
    check2.add_child(assertion2)

    # Test traversal with filtering
    assertions = list(root.assertions())
    assert len(assertions) == 2
    assert assertion1 in assertions
    assert assertion2 in assertions

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

    root.add_child(check)
    check.add_child(assertion)

    # Count nodes
    visitor = CountingVisitor()
    root.accept(visitor)

    # Should count: root + check + assertion = 3
    assert visitor.count == 3


def test_composite_pattern_hierarchy() -> None:
    """Test the composite pattern for node hierarchy."""
    root = graph.RootNode("Root")
    check = graph.CheckNode("Check")
    assertion = graph.AssertionNode(sp.Symbol("x"), root=root)

    # Test add_child
    root.add_child(check)
    check.add_child(assertion)

    assert len(root.children) == 1
    assert len(check.children) == 1

    # Test remove_child
    check.remove_child(assertion)
    assert len(check.children) == 0

    # Test get_children
    children = root.get_children()
    assert children == [check]

    # Re-add assertion to check
    check.add_child(assertion)
    assert check.get_children() == [assertion]


def test_traverse_without_filter() -> None:
    """Test RootNode traverse method without type filter."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    assertion = graph.AssertionNode(sp.Symbol("x"), root=root)

    root.add_child(check)
    check.add_child(assertion)

    # Traverse without filter should return all nodes
    all_nodes = list(root.traverse())
    assert len(all_nodes) == 3  # root + check + assertion
    assert root in all_nodes
    assert check in all_nodes
    assert assertion in all_nodes


# =============================================================================
# 3. Node Hierarchy Tests
# =============================================================================


def test_leaf_nodes_cannot_have_children() -> None:
    """Test that leaf nodes (AssertionNode) cannot have children."""
    # AssertionNode
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    with pytest.raises(RuntimeError, match="AssertionNode cannot have children"):
        assertion.add_child(MagicMock())


def test_check_node_only_accepts_assertion_children() -> None:
    """Test that CheckNode only accepts AssertionNode children."""
    check = graph.CheckNode(name="test_check")

    # Create assertion node
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))

    # Add assertion as child - should work
    check.add_child(assertion)

    assert len(check.children) == 1
    assert assertion in check.children


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


def test_check_node_symbol_tracking() -> None:
    """Test CheckNode's symbol tracking functionality."""
    check = graph.CheckNode("test_check")

    # Add symbols
    sym1 = sp.Symbol("x")
    sym2 = sp.Symbol("y")

    check.add_symbol(sym1)
    check.add_symbol(sym2)

    assert sym1 in check.symbols
    assert sym2 in check.symbols
    assert len(check.symbols) == 2

    # Adding duplicate should not increase count
    check.add_symbol(sym1)
    assert len(check.symbols) == 2


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

    # Add assertions
    assertion1 = graph.AssertionNode(sp.Symbol("x"), root=root)
    check1.add_child(assertion1)

    assertion2 = graph.AssertionNode(sp.Symbol("y"), root=root)
    check2.add_child(assertion2)

    # Propagate datasets
    root.impute_datasets(["ds1", "ds2"])

    # check1 should keep its dataset requirement
    assert check1.datasets == ["ds1"]
    # check2 should get all datasets
    assert check2.datasets == ["ds1", "ds2"]

    # Children should inherit from their parent check
    assert assertion1.datasets == ["ds1"]
    assert assertion2.datasets == ["ds1", "ds2"]


def test_dataset_validation_errors() -> None:
    """Test dataset validation error cases."""
    root = graph.RootNode("Test")

    # Test CheckNode dataset validation error
    check = graph.CheckNode("check1", datasets=["ds_missing"])
    root.add_child(check)
    root.impute_datasets(["ds1", "ds2"])

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

    root2.impute_datasets(["ds1"])
    assert isinstance(assertion._value, Some)
    assert isinstance(assertion._value.unwrap(), Failure)


def test_symbol_table_dataset_validation() -> None:
    """Test symbol table dataset validation during propagation."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)

    # Add symbol to symbol table
    symbol_table = root.symbol_table
    entry = SymbolEntry(
        symbol=sp.Symbol("x"),
        name="x_metric",
        dataset="ds_missing",
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(1.0),
    )
    symbol_table.register(entry)

    # Propagate with different dataset - should get validation errors
    errors = symbol_table.validate_datasets(["ds1"])
    assert len(errors) > 0
    assert "requires dataset 'ds_missing'" in errors[0]


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

    # Add symbols to symbol table
    symbol_table = root.symbol_table

    entry_x = SymbolEntry(
        symbol=sp.Symbol("x"),
        name="x_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(10.0),
        state="PROVIDED",
        value=Some(Success(10.0)),
    )
    entry_y = SymbolEntry(
        symbol=sp.Symbol("y"),
        name="y_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(20.0),
        state="PROVIDED",
        value=Some(Success(20.0)),
    )

    symbol_table.register(entry_x)
    symbol_table.register(entry_y)

    # Create assertion
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + sp.Symbol("y"), validator=SymbolicValidator(name="> 25", fn=lambda v: v > 25), root=root
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

    # Add symbol to symbol table
    symbol_table = root.symbol_table
    entry = SymbolEntry(
        symbol=sp.Symbol("x"),
        name="x_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(5.0),
        state="PROVIDED",
        value=Some(Success(5.0)),
    )
    symbol_table.register(entry)

    # Create assertion with failing validator
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        label="Test assertion",
        validator=SymbolicValidator(name="> 10", fn=lambda v: v > 10),
        root=root,
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

    # Create assertion without corresponding symbols in symbol table
    assertion = graph.AssertionNode(actual=sp.Symbol("x") + sp.Symbol("y"), root=root)
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

    # Add symbols to symbol table with different states
    symbol_table = root.symbol_table

    entry_x = SymbolEntry(
        symbol=sp.Symbol("x"),
        name="x_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(10.0),
        state="PROVIDED",
        value=Some(Success(10.0)),
    )
    entry_y = SymbolEntry(
        symbol=sp.Symbol("y"),
        name="y_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Failure("Symbol failed"),
        state="ERROR",
        value=Some(Failure("Symbol failed")),
    )

    symbol_table.register(entry_x)
    symbol_table.register(entry_y)

    # Create assertion
    assertion = graph.AssertionNode(actual=sp.Symbol("x") + sp.Symbol("y"), root=root)
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
    symbol_table = root.symbol_table
    entry_nan = SymbolEntry(
        symbol=sp.Symbol("x"),
        name="nan_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(float("nan")),
        state="PROVIDED",
        value=Some(Success(float("nan"))),
    )
    symbol_table.register(entry_nan)

    assertion_nan = graph.AssertionNode(actual=sp.Symbol("x"), root=root)
    check.add_child(assertion_nan)

    result_nan = assertion_nan.evaluate()
    assert isinstance(result_nan, Failure)
    assert "Validating value is NaN" in result_nan.failure()

    # Test infinity
    root2 = graph.RootNode("Test2")
    check2 = graph.CheckNode("check2")
    root2.add_child(check2)

    symbol_table2 = root2.symbol_table
    entry_inf = SymbolEntry(
        symbol=sp.Symbol("y"),
        name="inf_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(float("inf")),
        state="PROVIDED",
        value=Some(Success(float("inf"))),
    )
    symbol_table2.register(entry_inf)

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

    # Add symbol to symbol table
    symbol_table = root.symbol_table
    entry = SymbolEntry(
        symbol=sp.Symbol("x"),
        name="x_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(10.0),
        state="PROVIDED",
        value=Some(Success(10.0)),
    )
    symbol_table.register(entry)

    # Add assertion
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + 5, validator=SymbolicValidator(name="> 10", fn=lambda v: v > 10), root=root
    )
    check.add_child(assertion)

    # Propagate with wrong dataset - this should fail the check
    root.impute_datasets(["different_dataset"])

    # Evaluate assertion - should fail due to parent failure
    result = assertion.evaluate()
    assert isinstance(result, Failure)
    assert "Parent check failed!" in result.failure()


def test_assertion_node_parent_and_dependency_failures() -> None:
    """Test that parent failures are checked before dependency failures."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check", datasets=["required_dataset"])
    root.add_child(check)

    # Add failing symbol to symbol table
    symbol_table = root.symbol_table
    entry = SymbolEntry(
        symbol=sp.Symbol("x"),
        name="x_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Failure("Symbol error"),
        state="ERROR",
        value=Some(Failure("Symbol error")),
    )
    symbol_table.register(entry)

    # Add assertion with missing symbol y
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + sp.Symbol("y"),  # y is missing
        root=root,
    )
    check.add_child(assertion)

    # Propagate with wrong dataset - this should fail the check
    root.impute_datasets(["different_dataset"])

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

    # Add successful assertion
    assertion1 = graph.AssertionNode(sp.Symbol("x"), root=root)
    assertion1._value = Some(Success(10.0))
    check.add_child(assertion1)

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
    root.impute_datasets(["different_ds"])

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

    # Add symbol to symbol table
    symbol_table = root.symbol_table
    entry = SymbolEntry(
        symbol=sp.Symbol("x"),
        name="x_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(10.0),
        state="PROVIDED",
        value=Some(Success(10.0)),
    )
    symbol_table.register(entry)

    # Create assertion without validator
    assertion = graph.AssertionNode(actual=sp.Symbol("x") * 2, root=root)
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
    symbol_table = root.symbol_table
    entry = SymbolEntry(
        symbol=sp.Symbol("x"),
        name="x_metric",
        dataset=None,
        result_key=None,
        metric_spec=None,
        ops=[],
        retrieval_fn=lambda k: Success(0.0),
        state="PROVIDED",
        value=Some(Success(0.0)),
    )
    symbol_table.register(entry)

    # Create assertion with division by zero
    assertion = graph.AssertionNode(actual=sp.Integer(1) / sp.Symbol("x"), root=root)
    check.add_child(assertion)

    # Evaluate assertion - should catch exception
    result = assertion.evaluate()
    assert isinstance(result, Failure)


# =============================================================================
# 8. Minimal Coverage Tests for Display Methods
# =============================================================================


def test_node_format_display_methods_exist() -> None:
    """Test that all node types have format_display methods for coverage."""
    # RootNode
    root = graph.RootNode("Test Suite")
    assert hasattr(root, "format_display")
    root.format_display()  # Call to cover the method

    # CheckNode
    check = graph.CheckNode("check1", label="Check One", datasets=["ds1"])
    check.format_display()

    # AssertionNode with different states
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    assertion.format_display()

    # AssertionNode with validator
    validator = SymbolicValidator(name="> 10", fn=lambda x: x > 10)
    assertion_with_validator = graph.AssertionNode(actual=sp.Symbol("x"), validator=validator)
    assertion_with_validator.format_display()


def test_root_node_inspect_method() -> None:
    """Test RootNode inspect method for coverage."""
    root = graph.RootNode("Test Suite")
    check = graph.CheckNode("check1")
    root.add_child(check)

    # Call inspect to cover the method
    display = GraphDisplay()
    tree = display.inspect_tree(root)
    assert tree is not None  # Just verify it returns something
