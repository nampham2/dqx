"""Tests for the display module."""

from unittest.mock import Mock, patch

import sympy as sp

from dqx.common import SymbolicValidator
from dqx.display import print_graph
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph


class TestPrintGraph:
    """Test the print_graph function."""

    @patch("dqx.display.Console")
    def test_print_graph_basic(self, mock_console_class: Mock) -> None:
        """Test print_graph creates and prints a tree."""
        # Given a mock console and graph
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        # Create a simple graph structure
        root = RootNode("test_suite")
        check = root.add_check("Test Check")

        # Add assertions with validators
        positive_validator = SymbolicValidator("> 0", lambda x: x > 0)
        check.add_assertion(actual=sp.Symbol("x") > 0, name="Positive values", validator=positive_validator)

        graph = Graph(root)

        # When calling print_graph
        print_graph(graph)

        # Then console.print should be called with a Tree
        mock_console_class.assert_called_once()
        mock_console.print.assert_called_once()

    @patch("dqx.display.Console")
    def test_print_graph_with_multiple_checks(self, mock_console_class: Mock) -> None:
        """Test print_graph with multiple checks and assertions."""
        # Given a graph with multiple checks
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        root = RootNode("Quality Suite")

        # First check with assertions
        check1 = root.add_check("Data Completeness")
        positive_validator = SymbolicValidator("> 0", lambda x: x > 0)
        check1.add_assertion(actual=sp.Symbol("count") > 0, name="Has records", validator=positive_validator)

        # Second check
        check2 = root.add_check("Data Validity")
        upper_bound_validator = SymbolicValidator("< 100", lambda x: x < 100)
        check2.add_assertion(actual=sp.Symbol("x") < 100, name="Upper bound check", validator=upper_bound_validator)

        graph = Graph(root)

        # When printing
        print_graph(graph)

        # Then console.print should be called
        mock_console.print.assert_called_once()

    @patch("dqx.display.Console")
    @patch("dqx.display.Tree")
    def test_print_graph_tree_structure(self, mock_tree_class: Mock, mock_console_class: Mock) -> None:
        """Test that print_graph builds correct tree structure."""
        # Given mocks to inspect tree building
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        mock_tree = Mock()
        mock_tree_class.return_value = mock_tree

        # Mock tree.add to return new mock branches
        mock_check_branch = Mock()
        mock_tree.add.return_value = mock_check_branch

        # Create graph
        root = RootNode("Test Suite")
        check = root.add_check("Test Check")
        positive_validator = SymbolicValidator("> 0", lambda x: x > 0)
        check.add_assertion(actual=sp.Symbol("x") > 0, name="Test assertion", validator=positive_validator)

        graph = Graph(root)

        # When printing
        print_graph(graph)

        # Then tree should be created with root name
        mock_tree_class.assert_called_once()
        tree_label = mock_tree_class.call_args[0][0]
        assert "Test Suite" in tree_label

        # And check should be added to tree
        mock_tree.add.assert_called()
        check_label = mock_tree.add.call_args[0][0]
        assert "Test Check" in check_label

        # And assertion should be added to check branch
        mock_check_branch.add.assert_called()
        assertion_label = mock_check_branch.add.call_args[0][0]
        assert "Test assertion" in assertion_label

    @patch("dqx.display.Console")
    def test_print_graph_empty_checks(self, mock_console_class: Mock) -> None:
        """Test print_graph with root but no checks."""
        # Given a graph with only root
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        root = RootNode("Empty Suite")
        graph = Graph(root)

        # When printing
        print_graph(graph)

        # Then console.print should still be called
        mock_console.print.assert_called_once()

    @patch("dqx.display.Console")
    def test_print_graph_assertion_formatting(self, mock_console_class: Mock) -> None:
        """Test assertion label formatting in print_graph."""
        # Given a graph with various assertion types
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        root = RootNode("Test Suite")
        check = root.add_check("Assertions Test")

        # Assertion with validator
        validator_with_name = SymbolicValidator("is positive", lambda x: x > 0)
        check.add_assertion(actual=sp.Symbol("x") > 0, name="With validator", validator=validator_with_name)

        # Assertion with a no-op validator (instead of None)
        noop_validator = SymbolicValidator("", lambda x: True)
        check.add_assertion(actual=sp.Symbol("y") < 100, name="Without validator", validator=noop_validator)

        graph = Graph(root)

        # When printing
        print_graph(graph)

        # Then console.print should be called
        mock_console.print.assert_called_once()


class TestIntegration:
    """Integration tests with real graph structure."""

    @patch("dqx.display.Console")
    def test_real_graph_printing(self, mock_console_class: Mock) -> None:
        """Test full integration of print_graph with real nodes."""
        # Given a console mock
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        # Create a real graph
        root = RootNode("E-commerce Quality")

        # Add order validation check
        order_check = root.add_check("Order Validation")
        orders_exist_validator = SymbolicValidator("> 0", lambda x: x > 0)
        order_check.add_assertion(
            actual=sp.Symbol("order_count") > 0, name="Orders exist", validator=orders_exist_validator
        )

        # Add revenue check
        revenue_check = root.add_check("Revenue Check")
        positive_revenue_validator = SymbolicValidator(">= 0", lambda x: x >= 0)
        revenue_check.add_assertion(
            actual=sp.Symbol("revenue") >= 0, name="Non-negative revenue", validator=positive_revenue_validator
        )

        graph = Graph(root)

        # When printing
        print_graph(graph)

        # Then verify print was called
        mock_console.print.assert_called_once()

    def test_print_method_on_graph(self) -> None:
        """Test that Graph.print_tree delegates to print_graph."""
        # Given a graph
        root = RootNode("Test Suite")
        graph = Graph(root)

        # Mock print_graph
        with patch("dqx.display.print_graph") as mock_print_graph:
            # When calling print_tree
            graph.print_tree()

            # Then print_graph should be called with the graph only
            mock_print_graph.assert_called_once_with(graph)

        # Test with custom formatter - should issue a warning
        with patch("dqx.display.print_graph") as mock_print_graph:
            with patch("warnings.warn") as mock_warn:
                formatter = Mock()
                graph.print_tree(formatter=formatter)

                # Then print_graph should be called without formatter
                mock_print_graph.assert_called_once_with(graph)
                # And a warning should be issued
                mock_warn.assert_called_once_with(
                    "formatter argument is not supported by print_graph and will be ignored"
                )
