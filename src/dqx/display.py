"""Display module for graph visualization using Rich."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from returns.result import Result
from rich.console import Console
from rich.tree import Tree

if TYPE_CHECKING:
    from dqx.common import AssertionResult, EvaluationFailure
    from dqx.graph.base import BaseNode
    from dqx.graph.traversal import Graph
    from dqx.provider import SymbolInfo

# Type aliases for clarity
if TYPE_CHECKING:
    MetricValue = Result[float, list[EvaluationFailure]]
    SymbolValue = Result[float, str]


class NodeFormatter(Protocol):
    """Protocol for formatting graph nodes for display."""

    def format_node(self, node: BaseNode) -> str:
        """Format a node for display.

        Args:
            node: The node to format.

        Returns:
            A string representation of the node.
        """
        ...


class SimpleNodeFormatter:
    """Simple formatter that displays node label, name, or class name."""

    def format_node(self, node: BaseNode) -> str:
        """Format a node using priority: node_name() -> label -> name -> class name.

        Args:
            node: The node to format.

        Returns:
            A string representation of the node.
        """
        # First check for node_name() method (CheckNode has this)
        if hasattr(node, "node_name") and callable(node.node_name):
            return node.node_name()

        # Then check for label attribute
        if hasattr(node, "label") and node.label:
            return node.label

        # Then check for name attribute
        if hasattr(node, "name") and node.name:
            return node.name

        # Finally, use class name
        return node.__class__.__name__


class TreeBuilderVisitor:
    """Visitor that builds a Rich Tree during graph traversal."""

    def __init__(self, formatter: NodeFormatter) -> None:
        """Initialize the visitor with a node formatter.

        Args:
            formatter: The formatter to use for node labels.
        """
        self._formatter = formatter
        self.tree: Tree | None = None

        # Store the corresponding tree parent of a graph node
        # so that we can add children correctly
        self.parent_map: dict[BaseNode, Tree] = {}

    def visit(self, node: BaseNode) -> None:
        """Visit a node and add it to the tree.

        Args:
            node: The node to visit.

        Raises:
            ValueError: If the node's parent was not visited before the child.
        """
        formatted_label = self._formatter.format_node(node)

        # This is the root node
        if node.is_root:
            self.tree = Tree(formatted_label)
            self.parent_map[node] = self.tree
        else:
            # Find parent's tree node
            if node.parent not in self.parent_map:
                raise ValueError(
                    f"Parent of node '{formatted_label}' was not visited before the child. "
                    f"This indicates an issue with the traversal order."
                )

            parent_tree = self.parent_map[node.parent]
            child_tree = parent_tree.add(formatted_label)
            self.parent_map[node] = child_tree

    async def visit_async(self, node: BaseNode) -> None:
        """Async visit method that delegates to sync visit.

        Args:
            node: The node to visit.
        """
        self.visit(node)


def print_graph(graph: Graph, formatter: NodeFormatter | None = None) -> None:
    """Print a graph structure as a tree to the console.

    Args:
        graph: The graph to print.
        formatter: Optional formatter for node labels. Defaults to SimpleNodeFormatter.
    """
    formatter = formatter or SimpleNodeFormatter()

    visitor = TreeBuilderVisitor(formatter)
    graph.dfs(visitor)

    if visitor.tree is not None:
        console = Console()
        console.print(visitor.tree)


def print_assertion_results(results: list[AssertionResult]) -> None:
    """
    Display assertion results in a formatted table.

    Shows all fields from AssertionResult objects in a table with columns:
    Date, Suite, Check, Assertion, Expression, Severity, Status, Value/Error, Tags

    Args:
        results: List of AssertionResult objects from collect_results()

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.run(datasources, key)
        >>> results = suite.collect_results()
        >>> print_assertion_results(results)
    """
    from returns.result import Failure, Success
    from rich.table import Table

    # Create table with title
    table = Table(title="Assertion Results", show_lines=True)

    # Add columns in specified order
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Suite", style="blue")
    table.add_column("Check", style="yellow")
    table.add_column("Assertion")
    table.add_column("Expression", style="dim")
    table.add_column("Severity", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Value/Error")
    table.add_column("Tags", style="dim")

    # Define severity colors
    severity_colors = {"P0": "red", "P1": "yellow", "P2": "blue", "P3": "dim"}

    # Add rows
    for result in results:
        # Format status with color
        status_style = "green bold" if result.status == "OK" else "red bold"
        status_display = f"[{status_style}]{result.status}[/{status_style}]"

        # Format severity with color
        severity_color = severity_colors.get(result.severity, "white")
        severity_display = f"[{severity_color}]{result.severity}[/{severity_color}]"

        # Extract value/error using pattern matching with colors
        match result.metric:
            case Success(value):
                value_display = f"[green]{value}[/green]"
            case Failure(failures):
                error_text = "; ".join(f.error_message for f in failures)
                value_display = f"[red]{error_text}[/red]"

        # Format tags as key=value pairs
        tags_display = ", ".join(f"{k}={v}" for k, v in result.tags.items())
        if not tags_display:
            tags_display = "-"

        # Add row
        table.add_row(
            result.yyyy_mm_dd.isoformat(),
            result.suite,
            result.check,
            result.assertion,
            result.expression or "-",
            severity_display,
            status_display,
            value_display,
            tags_display,
        )

    # Print table
    console = Console()
    console.print(table)


def print_symbols(symbols: list[SymbolInfo]) -> None:
    """
    Display symbol values in a formatted table.

    Shows all fields from SymbolInfo objects in a table with columns:
    Date, Symbol, Metric, Dataset, Value/Error, Tags

    Args:
        symbols: List of SymbolInfo objects from collect_symbols()

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.run(datasources, key)
        >>> symbols = suite.collect_symbols()
        >>> print_symbols(symbols)
    """
    from returns.result import Failure, Success
    from rich.table import Table

    # Create table with title
    table = Table(title="Symbol Values", show_lines=True)

    # Add columns in specified order
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Symbol", style="yellow", no_wrap=True)
    table.add_column("Metric")
    table.add_column("Dataset", style="magenta")
    table.add_column("Value/Error")
    table.add_column("Tags", style="dim")

    # Flat display of symbols
    for symbol in symbols:
        # Extract value/error using pattern matching with colors
        match symbol.value:
            case Success(value):
                value_display = f"[green]{value}[/green]"
            case Failure(error):
                value_display = f"[red]{error}[/red]"

        # Format tags as key=value pairs
        tags_display = ", ".join(f"{k}={v}" for k, v in symbol.tags.items())
        if not tags_display:
            tags_display = "-"

        # Add row
        table.add_row(
            symbol.yyyy_mm_dd.isoformat(),
            symbol.name,
            symbol.metric,
            symbol.dataset or "-",
            value_display,
            tags_display,
        )

    # Print table
    console = Console()
    console.print(table)
