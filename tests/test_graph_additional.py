"""Additional tests to achieve 100% coverage for graph module."""


from typing import Any

import pytest
import sympy as sp
from returns.maybe import Some
from returns.result import Failure, Success

from dqx import graph


def test_format_error_exception_cases() -> None:
    """Test edge cases in _format_error that trigger exceptions."""
    from dqx.graph import _format_error
    
    # Test case that triggers IndexError in dataset parsing
    # This happens when the split doesn't produce expected parts
    msg_with_no_space = "requires datasets[]but got[]"
    result = _format_error(msg_with_no_space)
    assert "[red]❌" in result
    assert msg_with_no_space in result
    
    # Test case that triggers AttributeError in validation parsing
    # This could happen if message.split returns None somehow (edge case)
    class BadString(str):
        def split(self, sep: str | None = None, maxsplit: int = -1) -> list[str]:  # type: ignore[override]
            if sep == ": ":
                raise AttributeError("Bad split")
            return super().split(sep, maxsplit)
    
    bad_msg = BadString("Something: x = 5 does not satisfy > 10")
    result = _format_error(bad_msg)
    # Should fall through to default formatting
    assert "[red]❌ Something: x = 5 does not satisfy > 10[/red]" == result


def test_format_error_edge_case_with_colon_but_no_parts() -> None:
    """Test _format_error with messages that have colons but don't split properly."""
    from dqx.graph import _format_error
    
    # Message with "does not satisfy" but no colon to split on
    msg = "does not satisfy > 10"
    result = _format_error(msg)
    assert "[red]❌ does not satisfy > 10[/red]" == result
    
    # Message with colon but splits into "does not satisfy" part
    msg = ": does not satisfy"
    result = _format_error(msg)
    # The code splits on ":" and takes the last part with leading space
    assert "[red]❌  does not satisfy[/red]" == result or "[red]❌ does not satisfy[/red]" == result


def test_symbol_node_inspect_str_edge_cases() -> None:
    """Test edge cases in SymbolNode inspect_str formatting."""
    # Test with exact 1000 value (boundary case)
    symbol = graph.SymbolNode("metric", sp.Symbol("x"), lambda k: Success(1000.0), [])
    symbol._value = Some(Success(1000.0))
    
    inspect_str = symbol.inspect_str()
    # At exactly 1000, it should use str() which shows "1000.0" or "1000"
    assert "x: metric = 1000" in inspect_str or "x: metric = 1000.0" in inspect_str


def test_assertion_node_inspect_str_float_integer_edge_case() -> None:
    """Test AssertionNode inspect_str with floats that are integers."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create a symbol with a float that's actually an integer
    symbol = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Success(10.0), [])
    check.add_child(symbol)
    symbol._value = Some(Success(10.0))
    
    # Create assertion with validator
    validator = graph.SymbolicValidator(name="> 5", fn=lambda x: x > 5)
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=validator,
        root=root
    )
    check.add_child(assertion)
    
    # Evaluate to make it successful
    assertion.evaluate()
    inspect_str = assertion.inspect_str()
    
    # Should format as integer "10" not "10.0" 
    assert "(10)" in inspect_str


def test_missing_coverage_lines() -> None:
    """Test to cover any remaining missing lines."""
    # Test SymbolNode with failed value formatting
    symbol = graph.SymbolNode("metric", sp.Symbol("x"), lambda k: Failure("error"), [])
    symbol._value = Some(Failure("error"))
    
    # Should not include value in inspect_str when failed
    inspect_str = symbol.inspect_str()
    assert " = " not in inspect_str  # No value shown for failures
    assert "❌" in inspect_str


def test_assertion_inspect_str_exception_handling() -> None:
    """Test exception handling in AssertionNode inspect_str method."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create assertion with validator
    validator = graph.SymbolicValidator(name="> 10", fn=lambda x: x > 10)
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=validator,
        root=root
    )
    check.add_child(assertion)
    
    # Create a Failure with a message that has " = " but no " does not" after it
    # This will cause IndexError when trying to split on " does not" and access [0]
    assertion._value = Some(Failure("Test: x = "))
    
    # This should trigger the IndexError exception handler
    inspect_str = assertion.inspect_str()
    assert "Test: x = " in inspect_str
    
    
def test_assertion_inspect_str_attribute_error() -> None:
    """Test edge case in AssertionNode formatting."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create assertion with validator
    validator = graph.SymbolicValidator(name="> 10", fn=lambda x: x > 10)
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=validator,
        root=root
    )
    check.add_child(assertion)
    
    # Set a Failure message that will trigger default formatting
    assertion._value = Some(Failure("some other error message"))
    
    # This should trigger the else block
    inspect_str = assertion.inspect_str()
    assert "some other error message" in inspect_str


def test_assertion_node_inspect_without_is_integer() -> None:
    """Test AssertionNode inspect_str with values that don't have is_integer method."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create assertion with validator
    validator = graph.SymbolicValidator(name="> 5", fn=lambda x: x > 5)
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=validator,
        root=root
    )
    check.add_child(assertion)
    
    # Create a successful result with an integer
    assertion._value = Some(Success(10))  # Plain integer
    
    inspect_str = assertion.inspect_str()
    # Should format as integer "10"
    assert "(10)" in inspect_str
    assert "✓" in inspect_str


def test_assertion_find_root_runtime_error() -> None:
    """Test _find_root raises RuntimeError when root is None."""
    # Create assertion without root
    validator = graph.SymbolicValidator(name="> 10", fn=lambda x: x > 10)
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=validator,
        root=None
    )
    
    # Try to call _find_root, should raise RuntimeError
    with pytest.raises(RuntimeError, match="Root node not set for AssertionNode"):
        assertion._find_root()


def test_assertion_node_inspect_str_non_integer_float() -> None:
    """Test AssertionNode inspect_str with non-integer float values."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create a symbol with a non-integer float value
    symbol = graph.SymbolNode("x_metric", sp.Symbol("x"), lambda k: Success(123.456), [])
    check.add_child(symbol)
    symbol._value = Some(Success(123.456))
    
    # Create assertion with validator
    validator = graph.SymbolicValidator(name="> 100", fn=lambda x: x > 100)
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=validator,
        root=root
    )
    check.add_child(assertion)
    
    # Evaluate to make it successful
    assertion.evaluate()
    inspect_str = assertion.inspect_str()
    
    # Should format as "123.46" (2 decimal places for values < 1000)
    assert "(123.46)" in inspect_str
    
    # Test with a large float value (>= 1000)
    symbol2 = graph.SymbolNode("y_metric", sp.Symbol("y"), lambda k: Success(1234.567), [])
    check.add_child(symbol2)
    symbol2._value = Some(Success(1234.567))
    
    assertion2 = graph.AssertionNode(
        actual=sp.Symbol("y"),
        validator=graph.SymbolicValidator(name="> 1000", fn=lambda x: x > 1000),
        root=root
    )
    check.add_child(assertion2)
    
    assertion2.evaluate()
    inspect_str2 = assertion2.inspect_str()
    
    # Should format as "1234.6" (1 decimal place for values >= 1000)
    assert "(1234.6)" in inspect_str2


def test_assertion_inspect_str_malformed_does_not_satisfy_with_exception() -> None:
    """Test edge cases in assertion error message parsing."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create assertion with validator
    validator = graph.SymbolicValidator(name="> 10", fn=lambda x: x > 10)
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=validator,
        root=root
    )
    check.add_child(assertion)
    
    # Test case 1: Message with " = " followed by "does not satisfy" without value
    # This tests the edge case where split produces empty value
    assertion._value = Some(Failure("Test: x = does not satisfy"))
    
    inspect_str = assertion.inspect_str()
    # When parsed, this gives empty value: "Value  exceeds limit" 
    assert "Value does not satisfy exceeds limit" in inspect_str or "Value  exceeds limit" in inspect_str
    
    # Test case 2: Message without proper "does not satisfy" pattern
    assertion._value = Some(Failure("x = invalid"))
    
    inspect_str = assertion.inspect_str()
    # Should show the full error since it doesn't match the pattern
    assert "x = invalid" in inspect_str


def test_assertion_node_inspect_str_without_validator() -> None:
    """Test AssertionNode inspect_str when validator is None."""
    # Create assertion without validator
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x") + sp.Symbol("y"),
        label="test label",
        root=None
    )
    
    # The inspect_str should just return the actual expression
    inspect_str = assertion.inspect_str()
    assert inspect_str == "x + y"


def test_analyzer_node_add_child_raises_error() -> None:
    """Test that AnalyzerNode.add_child raises NotImplementedError."""
    from dqx.ops import Op
    
    # Create a mock Op
    class MockOp(Op):
        @property
        def name(self) -> str:
            return "MockAnalyzer"
        
        @property
        def prefix(self) -> str:
            return "mock"
        
        def value(self) -> Any:
            return "value"
        
        def assign(self, value: Any) -> None:
            pass
        
        def clear(self) -> None:
            pass
    
    # Create AnalyzerNode
    analyzer_node = graph.AnalyzerNode(MockOp())
    
    # Try to add a child - should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="AnalyzerNode cannot have children"):
        analyzer_node.add_child("some_child")


def test_assertion_find_root_returns_root() -> None:
    """Test _find_root returns the root when it exists."""
    root = graph.RootNode("Test")
    
    # Create assertion with root
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=None,
        root=root
    )
    
    # Call _find_root and verify it returns the root
    result = assertion._find_root()
    assert result is root
    assert isinstance(result, graph.RootNode)


def test_assertion_inspect_str_exception_in_error_parsing() -> None:
    """Test exception handling in AssertionNode inspect_str error parsing."""
    root = graph.RootNode("Test")
    check = graph.CheckNode("check")
    root.add_child(check)
    
    # Create assertion with validator
    validator = graph.SymbolicValidator(name="> 10", fn=lambda x: x > 10)
    assertion = graph.AssertionNode(
        actual=sp.Symbol("x"),
        validator=validator,
        root=root
    )
    check.add_child(assertion)
    
    # Create a custom string class that raises AttributeError when splitting
    class BadString(str):
        def split(self, sep: str | None = None, maxsplit: int = -1) -> list[str]:  # type: ignore[override]
            if sep == " = ":
                raise AttributeError("Bad split on equals")
            return super().split(sep, maxsplit)
    
    # Set a failure with our bad string
    assertion._value = Some(Failure(BadString("Test: x = 5 does not satisfy > 10")))
    
    # This should trigger the AttributeError handler at lines 545-546
    inspect_str = assertion.inspect_str()
    # Should fall back to showing the full error
    assert "Test: x = 5 does not satisfy > 10" in inspect_str
