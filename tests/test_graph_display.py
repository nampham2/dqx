"""Tests for Graph.print_tree method."""

from unittest.mock import Mock, patch
from dqx.graph.nodes import RootNode, CheckNode, AssertionNode
from dqx.graph.traversal import Graph
from dqx.display import NodeFormatter
import sympy as sp


class TestGraphPrintTree:
    """Test the print_tree method on Graph class."""

    @patch("dqx.display.print_graph")
    def test_print_tree_with_default_formatter(self, mock_print_graph: Mock) -> None:
        """Test that print_tree calls print_graph with default formatter."""
        # Given a graph
        root = RootNode("test_suite")
        graph = Graph(root)

        # When calling print_tree
        graph.print_tree()

        # Then print_graph should be called with graph and None formatter
        mock_print_graph.assert_called_once_with(graph, None)

    @patch("dqx.display.print_graph")
    def test_print_tree_with_custom_formatter(self, mock_print_graph: Mock) -> None:
        """Test that print_tree passes custom formatter to print_graph."""
        # Given a graph and custom formatter
        root = RootNode("test_suite")
        graph = Graph(root)
        custom_formatter = Mock(spec=NodeFormatter)

        # When calling print_tree with formatter
        graph.print_tree(formatter=custom_formatter)

        # Then print_graph should be called with graph and custom formatter
        mock_print_graph.assert_called_once_with(graph, custom_formatter)

    @patch("dqx.display.print_graph")
    def test_print_tree_with_complex_graph(self, mock_print_graph: Mock) -> None:
        """Test print_tree with a more complex graph structure."""
        # Given a complex graph
        root = RootNode("Quality Suite")
        check1 = CheckNode("completeness", label="Data Completeness")
        check2 = CheckNode("validity", label="Data Validity")

        assertion1 = AssertionNode(actual=sp.Symbol("x") > 0, label="Positive values")
        assertion2 = AssertionNode(actual=sp.Symbol("y") < 100, label="Upper bound check")

        root.add_child(check1)
        root.add_child(check2)
        check1.add_child(assertion1)
        check2.add_child(assertion2)

        graph = Graph(root)

        # When calling print_tree
        graph.print_tree()

        # Then print_graph should be called with the graph
        mock_print_graph.assert_called_once_with(graph, None)
