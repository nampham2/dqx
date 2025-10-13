"""Tests for Graph.print_tree method."""

from unittest.mock import Mock, patch

import sympy as sp

from dqx.common import SymbolicValidator
from dqx.display import NodeFormatter
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph


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
        check1 = root.add_check("Data Completeness")
        check2 = root.add_check("Data Validity")

        positive_validator = SymbolicValidator("> 0", lambda x: x > 0)
        upper_bound_validator = SymbolicValidator("< 100", lambda x: x < 100)
        check1.add_assertion(actual=sp.Symbol("x") > 0, name="Positive values", validator=positive_validator)
        check2.add_assertion(actual=sp.Symbol("y") < 100, name="Upper bound check", validator=upper_bound_validator)

        graph = Graph(root)

        # When calling print_tree
        graph.print_tree()

        # Then print_graph should be called with the graph
        mock_print_graph.assert_called_once_with(graph, None)
