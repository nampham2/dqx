"""Tests for the display module."""

from unittest.mock import Mock, patch

import pytest
import sympy as sp
from rich.tree import Tree

from dqx.common import SymbolicValidator
from dqx.display import NodeFormatter, SimpleNodeFormatter, TreeBuilderVisitor, print_graph
from dqx.graph.base import BaseNode
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph


class TestNodeFormatter:
    """Test the NodeFormatter protocol and SimpleNodeFormatter implementation."""

    def test_simple_formatter_with_label(self) -> None:
        """Test that formatter returns label when available."""
        # Given a node with a label attribute
        node = Mock(spec=BaseNode)
        node.label = "Test Label"
        node.name = "test_name"
        node.__class__.__name__ = "MockNode"

        # When formatting the node
        formatter = SimpleNodeFormatter()
        result = formatter.format_node(node)

        # Then it should return the label
        assert result == "Test Label"

    def test_simple_formatter_with_name_no_label(self) -> None:
        """Test that formatter returns name when label is not available."""
        # Given a node with only name attribute
        node = Mock(spec=BaseNode)
        node.label = None
        node.name = "test_name"
        node.__class__.__name__ = "MockNode"

        # When formatting the node
        formatter = SimpleNodeFormatter()
        result = formatter.format_node(node)

        # Then it should return the name
        assert result == "test_name"

    def test_simple_formatter_with_empty_label(self) -> None:
        """Test that formatter returns name when label is empty."""
        # Given a node with empty label
        node = Mock(spec=BaseNode)
        node.label = ""
        node.name = "test_name"
        node.__class__.__name__ = "MockNode"

        # When formatting the node
        formatter = SimpleNodeFormatter()
        result = formatter.format_node(node)

        # Then it should return the name
        assert result == "test_name"

    def test_simple_formatter_no_label_no_name(self) -> None:
        """Test that formatter returns class name when no label or name."""
        # Given a node without label or name attributes
        node = Mock(spec=BaseNode)
        node.__class__.__name__ = "MockNode"

        # When formatting the node
        formatter = SimpleNodeFormatter()
        result = formatter.format_node(node)

        # Then it should return the class name
        assert result == "MockNode"

    def test_simple_formatter_with_empty_name(self) -> None:
        """Test that formatter returns class name when name is empty."""
        # Given a node with empty name
        node = Mock(spec=BaseNode)
        node.label = None
        node.name = ""
        node.__class__.__name__ = "MockNode"

        # When formatting the node
        formatter = SimpleNodeFormatter()
        result = formatter.format_node(node)

        # Then it should return the class name
        assert result == "MockNode"

    def test_simple_formatter_with_node_name_method(self) -> None:
        """Test that formatter uses node_name() method when available (CheckNode)."""
        # Given a CheckNode with node_name method
        root = RootNode("test_suite")
        check = root.add_check("Check One")

        # When formatting the node
        formatter = SimpleNodeFormatter()
        result = formatter.format_node(check)

        # Then it should use node_name() which returns name
        assert result == "Check One"

    def test_simple_formatter_with_node_name_method_no_label(self) -> None:
        """Test that formatter uses node_name() method when no label (CheckNode)."""
        # Given a CheckNode without label
        root = RootNode("test_suite")
        check = root.add_check("check1")

        # When formatting the node
        formatter = SimpleNodeFormatter()
        result = formatter.format_node(check)

        # Then it should use node_name() which returns name
        assert result == "check1"


class TestTreeBuilderVisitor:
    """Test the TreeBuilderVisitor class."""

    def test_visitor_creates_root_tree(self) -> None:
        """Test that visitor creates root tree on first visit."""
        # Given a formatter and visitor
        formatter = SimpleNodeFormatter()
        visitor = TreeBuilderVisitor(formatter)

        # When visiting first node
        node = Mock(spec=BaseNode)
        node.parent = None
        node.__class__.__name__ = "RootNode"
        visitor.visit(node)

        # Then tree should be created
        assert visitor.tree is not None
        assert isinstance(visitor.tree, Tree)
        assert "RootNode" in str(visitor.tree.label)

    def test_visitor_adds_child_to_parent_tree(self) -> None:
        """Test that visitor correctly adds child nodes."""
        # Given a visitor with a root node already visited
        formatter = SimpleNodeFormatter()
        visitor = TreeBuilderVisitor(formatter)

        # Visit root
        root = Mock(spec=BaseNode)
        root.parent = None
        root.is_root = True
        root.__class__.__name__ = "RootNode"
        visitor.visit(root)

        # When visiting child node
        child = Mock(spec=BaseNode)
        child.parent = root
        child.is_root = False
        child.__class__.__name__ = "ChildNode"
        visitor.visit(child)

        # Then child should be in tree map
        assert child in visitor.parent_map
        # And root tree should have a child
        assert visitor.tree is not None
        assert len(visitor.tree.children) == 1

    def test_visitor_error_on_unvisited_parent(self) -> None:
        """Test that visitor raises error if parent not visited first."""
        # Given a visitor
        formatter = SimpleNodeFormatter()
        visitor = TreeBuilderVisitor(formatter)

        # First create a root to establish the tree
        root = Mock(spec=BaseNode)
        root.parent = None
        root.is_root = True
        root.__class__.__name__ = "RootNode"
        visitor.visit(root)

        # When visiting a node with unvisited parent
        parent = Mock(spec=BaseNode)
        child = Mock(spec=BaseNode)
        child.parent = parent
        child.is_root = False
        child.__class__.__name__ = "ChildNode"

        # Then should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            visitor.visit(child)

        assert "Parent of node" in str(exc_info.value)
        assert "was not visited before the child" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_visitor_async_delegates_to_visit(self) -> None:
        """Test that visit_async delegates to visit."""
        # Given a visitor with mocked visit method
        formatter = SimpleNodeFormatter()
        visitor = TreeBuilderVisitor(formatter)
        visitor.visit = Mock()  # type: ignore[method-assign]

        # When calling visit_async
        node = Mock(spec=BaseNode)
        await visitor.visit_async(node)

        # Then it should call visit
        visitor.visit.assert_called_once_with(node)


class TestPrintGraph:
    """Test the print_graph function."""

    @patch("dqx.display.Console")
    def test_print_graph_with_default_formatter(self, mock_console_class: Mock) -> None:
        """Test print_graph uses default formatter when none provided."""
        # Given a mock console and graph
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        # Create a mock node to simulate DFS traversal
        mock_node = Mock(spec=BaseNode)
        mock_node.parent = None
        mock_node.__class__.__name__ = "RootNode"

        # Mock graph.dfs to simulate traversal by calling visitor
        mock_graph = Mock(spec=Graph)

        def mock_dfs(visitor: TreeBuilderVisitor) -> None:
            visitor.visit(mock_node)

        mock_graph.dfs = Mock(side_effect=mock_dfs)

        # When calling print_graph
        print_graph(mock_graph)

        # Then it should create console and call dfs
        mock_console_class.assert_called_once()
        mock_graph.dfs.assert_called_once()

        # And console.print should be called with the tree
        mock_console.print.assert_called_once()

        # And the visitor should be TreeBuilderVisitor
        visitor_arg = mock_graph.dfs.call_args[0][0]
        assert isinstance(visitor_arg, TreeBuilderVisitor)
        assert isinstance(visitor_arg._formatter, SimpleNodeFormatter)

    @patch("dqx.display.Console")
    def test_print_graph_with_custom_formatter(self, mock_console_class: Mock) -> None:
        """Test print_graph uses provided formatter."""
        # Given a custom formatter
        custom_formatter = Mock(spec=NodeFormatter)
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        mock_graph = Mock(spec=Graph)
        mock_graph.dfs = Mock()

        # When calling print_graph with formatter
        print_graph(mock_graph, formatter=custom_formatter)

        # Then visitor should use custom formatter
        visitor_arg = mock_graph.dfs.call_args[0][0]
        assert isinstance(visitor_arg, TreeBuilderVisitor)
        assert visitor_arg._formatter is custom_formatter

    @patch("dqx.display.Console")
    def test_print_graph_prints_tree(self, mock_console_class: Mock) -> None:
        """Test that print_graph prints the built tree."""
        # Given a graph that will build a tree
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        # Create a simple graph structure
        root = RootNode("test_suite")
        root.add_check("Test Check")

        graph = Graph(root)

        # When calling print_graph
        print_graph(graph)

        # Then console.print should be called with a Tree
        mock_console.print.assert_called_once()
        printed_arg = mock_console.print.call_args[0][0]
        assert isinstance(printed_arg, Tree)

    @patch("dqx.display.Console")
    def test_print_graph_no_tree_no_print(self, mock_console_class: Mock) -> None:
        """Test that print_graph doesn't print if no tree built."""
        # Given an empty graph
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        mock_graph = Mock(spec=Graph)
        mock_graph.dfs = Mock()

        # When calling print_graph (visitor.tree remains None)
        print_graph(mock_graph)

        # Then console.print should not be called
        mock_console.print.assert_not_called()


class TestIntegration:
    """Integration tests with real node classes."""

    def test_real_graph_structure(self) -> None:
        """Test building tree with real graph nodes."""
        # Given a real graph structure
        root = RootNode("Quality Suite")

        check1 = root.add_check("Data Completeness")
        root.add_check("Data Validity")

        # Create assertions
        symbol = sp.Symbol("x")
        positive_validator = SymbolicValidator("> 0", lambda x: x > 0)
        upper_bound_validator = SymbolicValidator("< 100", lambda x: x < 100)
        check1.add_assertion(actual=symbol > 0, name="Positive values", validator=positive_validator)
        check1.add_assertion(actual=symbol < 100, name="Upper bound check", validator=upper_bound_validator)

        # Create visitor and traverse
        formatter = SimpleNodeFormatter()
        visitor = TreeBuilderVisitor(formatter)

        graph = Graph(root)
        graph.dfs(visitor)

        # Verify tree structure
        assert visitor.tree is not None
        assert "Quality Suite" in str(visitor.tree.label)
        assert len(visitor.tree.children) == 2

        # Check first check node
        check1_tree = visitor.tree.children[0]
        assert "Data Completeness" in str(check1_tree.label)
        assert len(check1_tree.children) == 2

        # Check assertions
        assert1_tree = check1_tree.children[0]
        assert "Positive values" in str(assert1_tree.label)

        assert2_tree = check1_tree.children[1]
        assert "Upper bound check" in str(assert2_tree.label)  # No label, so class name

    @patch("dqx.display.Console")
    def test_full_print_graph_integration(self, mock_console_class: Mock) -> None:
        """Test full integration of print_graph with real nodes."""
        # Given a console mock
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        # Create a real graph
        root = RootNode("E-commerce Quality")
        check = root.add_check("Order Validation")
        orders_exist_validator = SymbolicValidator("> 0", lambda x: x > 0)
        check.add_assertion(actual=sp.Symbol("order_count") > 0, name="Orders exist", validator=orders_exist_validator)

        graph = Graph(root)

        # When printing
        print_graph(graph)

        # Then verify print was called with correct tree
        mock_console.print.assert_called_once()
        tree = mock_console.print.call_args[0][0]

        assert isinstance(tree, Tree)
        assert "E-commerce Quality" in str(tree.label)
        assert len(tree.children) == 1
        assert "Order Validation" in str(tree.children[0].label)
