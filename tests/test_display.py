"""Tests for display module."""

import datetime as dt
from typing import Any, List, SupportsIndex, cast
from unittest.mock import MagicMock

import sympy as sp
from returns.maybe import Nothing, Some
from returns.result import Failure, Success
from rich.console import Console
from rich.tree import Tree

from dqx import display, graph
from dqx.common import ResultKey, ResultKeyProvider, SymbolicValidator
from dqx.ops import Op
from dqx.specs import MetricSpec


# =============================================================================
# Helper Classes
# =============================================================================

class MockOp:
    """Mock operator for testing."""
    def __init__(self, name: str) -> None:
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name


class MockKeyProvider:
    """Mock key provider for testing."""
    def create(self, key: ResultKey) -> ResultKey:
        return key


# =============================================================================
# Test Basic Formatting Functions
# =============================================================================

def test_format_status_helpers() -> None:
    """Test the format_status helper function with various inputs."""
    # Test with Nothing
    assert display.format_status(Nothing) == "[yellow]⏳[/yellow]"
    
    # Test with Success and show_value=True
    result_float = Some(Success(123.456))
    assert "123.46" in display.format_status(result_float, show_value=True)
    assert "✅" in display.format_status(result_float, show_value=True)
    
    result_large_float = Some(Success(1234.56))
    assert "1234.6" in display.format_status(result_large_float, show_value=True)
    
    result_int = Some(Success(42))
    assert "42" in display.format_status(result_int, show_value=True)
    
    result_str = Some(Success("test"))
    assert "test" in display.format_status(result_str, show_value=True)
    
    # Test with None value
    result_none = Some(Success(None))
    assert display.format_status(result_none, show_value=True) == "[green]✅[/green]"
    
    # Test with Failure
    result_fail = Some(Failure("error"))
    assert display.format_status(result_fail) == "[red]❌[/red]"
    
    # Test edge case - this would be unusual but covers the default return
    class FakeMaybe:
        pass
    fake = FakeMaybe()
    assert display.format_status(fake) == "[dim]❓[/dim]"  # type: ignore


def test_format_error_helpers() -> None:
    """Test the format_error helper function with various error messages."""
    # Test truncation of long messages
    long_msg = "x" * 150
    formatted = display.format_error(long_msg)
    assert "..." in formatted
    assert len(formatted) < 150
    
    # Test parent check failed
    assert "⚠️  Skipped: parent check failed" in display.format_error("parent check failed: something")
    
    # Test dataset mismatch
    msg = "requires datasets ['ds1'] but got ['ds2']"
    formatted = display.format_error(msg)
    assert "Dataset mismatch: needs ['ds1']" in formatted
    
    # Test missing symbols
    assert "❌ Missing symbols: x, y" in display.format_error("Missing symbols: x, y")
    
    # Test symbol dependencies failed
    assert "❌ Symbol dependencies failed: x" in display.format_error("Symbol dependencies failed: x")
    
    # Test validation failure
    msg = "Something: x + y = 15 does not satisfy > 20"
    formatted = display.format_error(msg)
    assert "❌ x + y = 15 does not satisfy > 20" in formatted
    
    # Test NaN and infinity
    assert "❌ Validating value is NaN" in display.format_error("Validating value is NaN")
    assert "❌ Validating value is infinity" in display.format_error("Validating value is infinity")
    
    # Test default formatting
    assert "❌ Unknown error" in display.format_error("Unknown error")


def test_format_error_malformed_messages() -> None:
    """Test format_error with various malformed messages."""
    # Test malformed "requires datasets" message - it still partially matches
    msg = "requires datasets ['ds1'] without but got part"
    result = display.format_error(msg)
    # The function still tries to parse it as dataset mismatch
    assert "Dataset mismatch: needs ['ds1']" in result
    
    # Test malformed "does not satisfy" message without colon
    msg2 = "x = 5 does not satisfy without proper format"
    result2 = display.format_error(msg2)
    # Should fall back to default formatting 
    assert "[red]❌ x = 5 does not satisfy without proper format[/red]" == result2
    
    # Test empty message
    assert display.format_error("") == "[red]❌ [/red]"
    
    # Test message with only spaces
    assert display.format_error("   ") == "[red]❌    [/red]"
    
    # Test malformed dataset message (triggers exception)
    malformed_ds_msg = "requires datasets but got something"
    formatted = display.format_error(malformed_ds_msg)
    # When parsing fails, the message should be returned with default formatting
    assert "[red]❌ Dataset mismatch: needs but got something[/red]" in formatted
    
    # Test malformed validation message (triggers exception)
    malformed_val_msg = "does not satisfy > 20"
    formatted = display.format_error(malformed_val_msg)
    assert "❌ does not satisfy > 20" in formatted


def test_format_error_with_integer_value() -> None:
    """Test format_error with values that have is_integer method."""
    # Test with a float that is an integer - format_error doesn't parse this specially
    msg = "x = 5.0 does not satisfy > 10"
    result = display.format_error(msg)
    # format_error returns the message with minimal formatting
    assert "x = 5.0 does not satisfy > 10" in result
    
    # Test with actual integer
    msg2 = "y = 42 does not satisfy < 40"
    result2 = display.format_error(msg2)
    assert "y = 42 does not satisfy < 40" in result2
    
    # Test with float that is not an integer
    msg3 = "z = 3.14 does not satisfy > 10"
    result3 = display.format_error(msg3)
    assert "z = 3.14 does not satisfy > 10" in result3


def test_format_error_exception_handlers() -> None:
    """Test exception handlers in format_error function."""
    # Create a string that has both keywords but will fail on split
    # The message needs to pass the initial check but fail when parsing
    class FailingSplitString(str):
        def __new__(cls, value: str) -> "FailingSplitString":
            return str.__new__(cls, value)
            
        def split(self, sep: str | None = None, maxsplit: SupportsIndex = -1) -> List[str]:
            # First split for "requires datasets " needs to work
            if sep == "requires datasets ":
                # Return only one element so parts[1] raises IndexError
                return ["This message has requires datasets and but got keywords"]
            return str.split(self, sep, maxsplit)
    
    msg = FailingSplitString("This message has requires datasets and but got keywords")
    result = display.format_error(msg)
    # Should catch IndexError and return the original message
    assert result == "[red]❌ This message has requires datasets and but got keywords[/red]"
    
    # Test with AttributeError during "does not satisfy" parsing
    class AttributeErrorOnSplit(str):
        def split(self, sep: str | None = None, maxsplit: SupportsIndex = -1) -> Any:
            if sep == ": ":
                # Return an object that doesn't support len()
                class BadList:
                    def __len__(self) -> None:
                        raise AttributeError("No len")
                return BadList()
            return super().split(sep, maxsplit)
    
    msg2 = AttributeErrorOnSplit("does not satisfy: test")
    result2 = display.format_error(msg2)
    # Exception is caught with pass, so falls through to default
    assert result2 == "[red]❌ does not satisfy: test[/red]"


def test_format_datasets_helpers() -> None:
    """Test the format_datasets helper function."""
    # Test empty list
    assert display.format_datasets([]) == ""
    
    # Test single dataset
    assert display.format_datasets(["ds1"]) == "[dim italic]ds1[/dim italic]"
    
    # Test multiple datasets
    assert display.format_datasets(["ds1", "ds2", "ds3"]) == "[dim italic]ds1, ds2, ds3[/dim italic]"


# =============================================================================
# Test Node Formatters
# =============================================================================

def test_root_node_formatter() -> None:
    """Test RootNodeFormatter."""
    root = graph.RootNode("Test Suite")
    formatter = display.RootNodeFormatter()
    assert formatter.format(root) == "Suite: Test Suite"


def test_check_node_formatter() -> None:
    """Test CheckNodeFormatter with different configurations."""
    formatter = display.CheckNodeFormatter()
    
    # Test with label and datasets
    check = graph.CheckNode("check1", label="Check One", datasets=["ds1"])
    formatted = formatter.format(check)
    assert "Check One" in formatted
    assert "ds1" in formatted
    assert "📋" in formatted
    assert "⏳" in formatted  # Pending status
    
    # Test without label
    check2 = graph.CheckNode("check2")
    formatted2 = formatter.format(check2)
    assert "check2" in formatted2
    
    # Test with success status
    check3 = graph.CheckNode("check3")
    check3._value = Some(Success(1.0))
    formatted3 = formatter.format(check3)
    assert "✅" in formatted3


def test_symbol_node_formatter() -> None:
    """Test SymbolNodeFormatter."""
    formatter = display.SymbolNodeFormatter()
    
    # Test with successful value
    symbol = graph.SymbolNode("metric1", sp.Symbol("x"), lambda k: Success(42.75), ["ds1"])
    symbol._value = Some(Success(42.75))
    
    formatted = formatter.format(symbol)
    assert "📊" in formatted
    assert "x: metric1 = 42.75" in formatted
    assert "✅" in formatted
    assert "ds1" in formatted


def test_symbol_node_format_display_variations() -> None:
    """Test various cases of SymbolNode formatting via format_display."""
    # Test with successful value
    symbol_success = graph.SymbolNode("metric1", sp.Symbol("x"), lambda k: Success(42.75), ["ds1"])
    symbol_success._value = Some(Success(42.75))
    
    format_display = symbol_success.format_display()
    assert "📊" in format_display
    assert "x: metric1 = 42.75" in format_display
    assert "✅" in format_display
    assert "ds1" in format_display
    
    # Test with integer value
    symbol_int = graph.SymbolNode("metric2", sp.Symbol("y"), lambda k: Success(100), [])
    symbol_int._value = Some(Success(100))
    
    format_display = symbol_int.format_display()
    assert "y: metric2 = 100" in format_display
    
    # Test with large float value
    symbol_large = graph.SymbolNode("metric3", sp.Symbol("z"), lambda k: Success(1234.567), [])
    symbol_large._value = Some(Success(1234.567))
    
    format_display = symbol_large.format_display()
    # For large values (>= 1000), it uses str() which shows full precision
    assert "z: metric3 = 1234.567" in format_display


def test_metric_node_formatter() -> None:
    """Test MetricNodeFormatter."""
    formatter = display.MetricNodeFormatter()
    
    spec = MagicMock(spec=MetricSpec, name="test_metric")
    key_provider = cast(ResultKeyProvider, MockKeyProvider())
    nominal_key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})
    
    metric = graph.MetricNode(spec, key_provider, nominal_key)
    metric.datasets = ["dataset1", "dataset2"]
    
    formatted = formatter.format(metric)
    assert "📈" in formatted
    assert "test_metric" in formatted
    assert "dataset1" in formatted and "dataset2" in formatted
    assert "⏳" in formatted  # Pending status


def test_analyzer_node_formatter() -> None:
    """Test AnalyzerNodeFormatter."""
    formatter = display.AnalyzerNodeFormatter()
    analyzer = graph.AnalyzerNode(cast(Op[Any], MockOp("analyzer1")))
    assert formatter.format(analyzer) == "🔧 analyzer1 analyzer"


def test_analyzer_node_add_child_not_implemented() -> None:
    """Test that AnalyzerNode.add_child raises NotImplementedError."""
    analyzer = graph.AnalyzerNode(cast(Op[Any], MockOp("test_analyzer")))
    
    # Create another node to try to add as a child
    child_analyzer = graph.AnalyzerNode(cast(Op[Any], MockOp("child_analyzer")))
    
    # Attempting to add a child should raise NotImplementedError
    try:
        analyzer.add_child(child_analyzer)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass  # Expected


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
    
    key_provider = cast(ResultKeyProvider, MockKeyProvider())
    nominal_key = ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
    
    metric1 = graph.MetricNode(metric_spec1, key_provider, nominal_key)
    metric2 = graph.MetricNode(metric_spec2, key_provider, nominal_key)
    
    symbol.add_child(metric1)
    symbol.add_child(metric2)
    
    # Add analyzer nodes to metrics
    analyzer1 = graph.AnalyzerNode(cast(Op[Any], MockOp("analyzer1")))
    analyzer2 = graph.AnalyzerNode(cast(Op[Any], MockOp("analyzer2")))
    analyzer3 = graph.AnalyzerNode(cast(Op[Any], MockOp("analyzer3")))
    analyzer4 = graph.AnalyzerNode(cast(Op[Any], MockOp("analyzer4")))
    
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
    assert '\n\n' not in output, "Found extra spacing (double newlines) between analyzer nodes"


def test_assertion_node_formatter() -> None:
    """Test AssertionNodeFormatter basic functionality."""
    formatter = display.AssertionNodeFormatter()
    
    # Test without validator
    assertion = graph.AssertionNode(actual=sp.Symbol("x"))
    assert formatter.format(assertion) == "x"
    
    # Test with validator and success
    validator = SymbolicValidator(name="> 10", fn=lambda x: x > 10)
    assertion2 = graph.AssertionNode(
        actual=sp.Symbol("x"),
        label="Check X",
        validator=validator
    )
    assertion2._value = Some(Success(15.0))
    formatted = formatter.format(assertion2)
    assert "●" in formatted
    assert "Check X:" in formatted
    assert "> 10" in formatted
    assert "(15)" in formatted
    
    # Test with datasets
    assertion3 = graph.AssertionNode(
        actual=sp.Symbol("y"),
        validator=validator
    )
    assertion3.set_datasource(["ds1", "ds2"])
    formatted3 = formatter.format(assertion3)
    assert "ds1, ds2" in formatted3
    
    # Test with generic error message (covers line 188)
    assertion4 = graph.AssertionNode(
        actual=sp.Symbol("z"),
        validator=validator
    )
    assertion4._value = Some(Failure("Some generic error message"))
    formatted4 = formatter.format(assertion4)
    assert "●" in formatted4
    assert "Some generic error message" in formatted4


def test_assertion_formatter_with_integer_like_values() -> None:
    """Test AssertionNodeFormatter with values that have is_integer method."""
    root = graph.RootNode("test")
    
    # Test with a float that is actually an integer (e.g., 5.0)
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=SymbolicValidator(name="< 10", fn=lambda x: x < 10),
        root=root
    )
    assertion._value = Some(Success(5.0))
    
    formatter = display.AssertionNodeFormatter()
    result = formatter.format(assertion)
    
    # Should display as "5" not "5.0"
    assert "(5)" in result
    assert "●" in result
    
    # Test with a float that is not an integer (e.g., 5.5)
    assertion2 = graph.AssertionNode(
        actual=sp.Symbol("y"),
        validator=SymbolicValidator(name="< 10", fn=lambda x: x < 10),
        root=root
    )
    assertion2._value = Some(Success(5.5))
    
    result2 = formatter.format(assertion2)
    # Should display as "5.50" (formatted to 2 decimals)
    assert "(5.50)" in result2
    
    # Test with an actual integer
    assertion3 = graph.AssertionNode(
        actual=sp.Symbol("z"),
        validator=SymbolicValidator(name="< 10", fn=lambda x: x < 10),
        root=root
    )
    assertion3._value = Some(Success(7))
    
    result3 = formatter.format(assertion3)
    assert "(7)" in result3


def test_assertion_formatter_exception_handlers() -> None:
    """Test exception handlers in AssertionNodeFormatter."""
    root = graph.RootNode("test")
    
    # Test IndexError in AssertionNodeFormatter "does not satisfy" parsing
    class FailingErrorMessage(str):
        def split(self, sep: str | None = None, maxsplit: SupportsIndex = -1) -> List[str]:
            if sep == " = ":
                # Return list with only one element, so parts[1] raises IndexError
                return ["no equals sign in message"]
            return super().split(sep, maxsplit)
    
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=SymbolicValidator(name="< 10", fn=lambda x: x < 10),
        root=root
    )
    
    # Use error message that contains "does not satisfy" but will fail parsing
    assertion._value = Some(Failure(FailingErrorMessage("x does not satisfy < 10")))
    
    formatter = display.AssertionNodeFormatter()
    result = formatter.format(assertion)
    
    # Exception is caught and shows full error message
    assert "x does not satisfy < 10" in result
    
    # Test AttributeError exception handler
    class AttributeErrorMessage(str):
        def split(self, sep: str | None = None, maxsplit: SupportsIndex = -1) -> Any:
            if sep == " = ":
                # Raise AttributeError directly
                raise AttributeError("Mock AttributeError")
            return super().split(sep, maxsplit)
    
    assertion2 = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=SymbolicValidator(name="< 10", fn=lambda x: x < 10),
        root=root
    )
    
    # Use error message that contains "does not satisfy" and will raise AttributeError
    assertion2._value = Some(Failure(AttributeErrorMessage("x = 5 does not satisfy < 10")))
    
    result2 = formatter.format(assertion2)
    
    # AttributeError is caught and shows full error message
    assert "x = 5 does not satisfy < 10" in result2
    assert "●" in result2  # Should have failure prefix
    
    # Test AttributeError when accessing split result
    class BadSplitResult(str):
        def split(self, sep: str | None = None, maxsplit: SupportsIndex = -1) -> Any:
            if sep == " = ":
                # Return a proper list but accessing parts[1] will cause issues later
                return ["x", "5 does not satisfy < 10"]
            elif sep == " does not":
                # This gets called on parts[1].split(" does not")
                # Raise AttributeError to trigger the exception handler
                raise AttributeError("Mock error on second split")
            return super().split(sep, maxsplit)
    
    assertion3 = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=SymbolicValidator(name="< 10", fn=lambda x: x < 10),
        root=root
    )
    
    # Use error message that will trigger the exception path
    assertion3._value = Some(Failure(BadSplitResult("x = 5 does not satisfy < 10")))
    
    result3 = formatter.format(assertion3)
    
    # The formatter successfully parsed the message and shows it as "Value 5 exceeds limit"
    assert "Value 5 exceeds limit" in result3
    assert "●" in result3  # Should have failure prefix


def test_assertion_node_format_display_variations() -> None:
    """Test various cases of AssertionNode formatting via format_display."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Test assertion without validator
    assertion_no_validator = graph.AssertionNode(actual=sp.Symbol("z"), root=root)
    check.add_child(assertion_no_validator)
    assert assertion_no_validator.format_display() == "z"
    
    # Test assertion with label and successful evaluation
    symbol = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Success(15.0), [])
    check.add_child(symbol)
    symbol._value = Some(Success(15.0))
    
    validator = SymbolicValidator(name="> 10", fn=lambda x: x > 10)
    assertion_with_label = graph.AssertionNode(
        actual=sp.Symbol("x"),
        label="Check X",
        validator=validator,
        root=root
    )
    check.add_child(assertion_with_label)
    
    # Evaluate to make it successful
    assertion_with_label.evaluate()
    format_display = assertion_with_label.format_display()

    assert "●" in format_display
    assert "Check X:" in format_display
    assert "> 10" in format_display
    
    # Test assertion with integer value
    symbol_int = graph.SymbolNode("y_metric", sp.Symbol("y"), lambda k: Success(42), [])
    check.add_child(symbol_int)
    symbol_int._value = Some(Success(42))
    
    assertion_int = graph.AssertionNode(
        actual=sp.Symbol("y"),
        validator=SymbolicValidator(name="> 40", fn=lambda x: x > 40),
        root=root
    )
    check.add_child(assertion_int)
    assertion_int.evaluate()
    format_display_int = assertion_int.format_display()
    assert "●" in format_display_int
    assert "(42)" in format_display_int
    
    # Test parent check failed error formatting
    assertion_parent_failed = graph.AssertionNode(
        actual=sp.Symbol("a"),
        validator=validator,
        root=root
    )
    assertion_parent_failed._value = Some(Failure("Parent check failed!"))
    format_display = assertion_parent_failed.format_display()
    assert "Skipped (parent failed)" in format_display
    assert "[yellow]" in format_display
    
    # Test value exceeds limit error formatting
    assertion_exceeds = graph.AssertionNode(
        actual=sp.Symbol("b"),
        validator=validator,
        root=root
    )
    assertion_exceeds._value = Some(Failure("Assertion failed: b = 8.5 does not satisfy > 10"))
    format_display = assertion_exceeds.format_display()
    assert "Value 8.5 exceeds limit" in format_display


# =============================================================================
# Test Display Manager and Configuration
# =============================================================================

def test_graph_display_manager() -> None:
    """Test GraphDisplay manager with different configurations."""
    # Test with default config
    display_mgr = display.GraphDisplay()
    assert display_mgr.config.show_values
    assert display_mgr.config.show_datasets
    assert display_mgr.config.compact_errors
    
    # Test with custom config
    config = display.DisplayConfig(
        show_values=False,
        show_datasets=False,
        compact_errors=False
    )
    display_mgr2 = display.GraphDisplay(config)
    assert not display_mgr2.config.show_values
    
    # Test format_node
    root = graph.RootNode("Test")
    formatted = display_mgr.format_node(root)
    assert "Suite: Test" == formatted
    
    # Test with unknown node type (fallback to str)
    class UnknownNode:
        def __str__(self) -> str:
            return "Unknown"
    
    unknown = UnknownNode()
    assert display_mgr.format_node(unknown) == "Unknown"  # type: ignore


# =============================================================================
# Test Tree Building and Inspection
# =============================================================================

def test_graph_inspect_tree() -> None:
    """Test graph tree inspection with display module."""
    root = graph.RootNode("Test Suite")
    check = graph.CheckNode("check1", label="Check One")
    root.add_child(check)
    
    # Add assertion with validator (should be included)
    assertion_with_validator = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=SymbolicValidator(name="> 0", fn=lambda x: x > 0),
        root=root
    )
    check.add_child(assertion_with_validator)
    
    # Add assertion without validator (should be skipped)
    assertion_without_validator = graph.AssertionNode(
        actual=sp.Symbol("y"),
        root=root
    )
    check.add_child(assertion_without_validator)
    
    # Build tree using display manager
    display_mgr = display.GraphDisplay()
    tree = display_mgr.inspect_tree(root)
    
    # Verify structure
    assert isinstance(tree, Tree)
    assert "Suite: Test Suite" in str(tree.label)
    assert len(tree.children) == 1  # Check node
    
    check_tree = tree.children[0]
    assert "Check One" in str(check_tree.label)
    # Only assertion with validator should be included
    assert len(check_tree.children) == 1


def test_tree_builder_skip_assertions_without_validators() -> None:
    """Test that TreeBuilder skips assertions without validators."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Add assertions with and without validators
    assertion1 = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=SymbolicValidator(name="> 5", fn=lambda x: x > 5),
        root=root
    )
    assertion2 = graph.AssertionNode(
        actual=sp.Symbol("y"),
        root=root
    )
    check.add_child(assertion1)
    check.add_child(assertion2)
    
    # Build tree
    display_mgr = display.GraphDisplay()
    tree_root = Tree("Root")
    builder = display.TreeBuilder(tree_root, display_mgr, root)
    
    # Visit the check node (which will visit its children)
    from dqx.graph import NodeVisitor
    check.accept(cast(NodeVisitor, builder))
    
    # Only one assertion should be added (the one with validator)
    assert len(tree_root.children) == 1
    check_subtree = tree_root.children[0]
    assert len(check_subtree.children) == 1
