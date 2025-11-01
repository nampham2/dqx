"""Display module for graph visualization using Rich."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

import pyarrow as pa
import sympy as sp
from returns.result import Result
from rich.console import Console
from rich.tree import Tree

from dqx.common import MetricKey

if TYPE_CHECKING:
    from dqx.analyzer import AnalysisReport
    from dqx.common import AssertionResult, EvaluationFailure
    from dqx.graph.base import BaseNode
    from dqx.graph.traversal import Graph
    from dqx.models import Metric
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


def print_metric_trace(trace_table: pa.Table, execution_id: str) -> None:
    """
    Display metric trace table with discrepancy highlighting.

    Shows how metric values flow from DB through analysis reports to final symbol values.
    Rows with differing values are highlighted to identify discrepancies (excluding extended metrics).
    The table is sorted by symbol indices (x_1, x_2, ..., x_10, ...) with non-symbol rows at the end.

    Extended metrics (DayOverDay, WeekOverWeek, Stddev) show green dashes for Value DB and
    Value Analysis columns to indicate they are computed metrics.

    Args:
        trace_table: PyArrow table from metric_trace() with columns:
                    date, metric, symbol, type, dataset, value_db, value_analysis, value_final,
                    error, tags, is_extended
        execution_id: The execution ID to display in the title

    Example:
        >>> trace = metric_trace(metrics, execution_id, reports, symbols)
        >>> print_metric_trace(trace, execution_id)
    """
    import re

    from rich.table import Table

    from dqx.data import metric_trace_stats

    # Get discrepancy statistics
    stats = metric_trace_stats(trace_table)

    # Create table with title
    table = Table(title=f"Data Trace for Execution: {execution_id}", show_lines=True)

    # Add columns (removed Type column)
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Metric", style="yellow", no_wrap=True)
    table.add_column("Symbol", style="yellow", no_wrap=True)
    table.add_column("Dataset", style="magenta")
    table.add_column("Value Analysis", style="dim")
    table.add_column("Value DB", style="dim")
    table.add_column("Value Final", style="dim")
    table.add_column("Error", style="red")
    table.add_column("Tags", style="dim")

    # Convert PyArrow table to dict
    data = trace_table.to_pydict()

    # Create a list of row indices for sorting
    row_indices = list(range(trace_table.num_rows))

    # Sort function to extract numeric part from symbol
    def symbol_sort_key(idx: int) -> tuple[int, int]:
        symbol = data["symbol"][idx]
        if symbol and symbol != "-":
            # Extract numeric part from x_N pattern
            match = re.match(r"x_(\d+)", symbol)
            if match:
                return (0, int(match.group(1)))  # (0, N) for symbols
        return (1, idx)  # (1, original_index) for non-symbols

    # Sort row indices by symbol
    row_indices.sort(key=symbol_sort_key)

    # Create a set of discrepancy row indices for quick lookup
    discrepancy_set = set(stats.discrepancy_rows)

    # Process each row in sorted order
    for i in row_indices:
        # Extract values
        value_db = data["value_db"][i]
        value_analysis = data["value_analysis"][i]
        value_final = data["value_final"][i]
        error = data["error"][i]
        is_extended = data["is_extended"][i] if "is_extended" in data else False

        # Check if this row has discrepancy
        has_discrepancy = i in discrepancy_set

        # Format values with colors
        def format_value(value: float | None, highlight: bool = False, is_extended: bool = False) -> str:
            if value is None:
                return "-"
            if highlight:
                return f"[bold red]{value}[/bold red]"
            return f"[green]{value}[/green]"

        value_db_display = format_value(value_db, has_discrepancy, is_extended)
        value_analysis_display = format_value(value_analysis, has_discrepancy, is_extended)
        value_final_display = format_value(value_final, has_discrepancy and not is_extended)

        # Format error - show green dash if no error
        error_display = "[green]-[/green]" if not error else error

        # Format other fields
        symbol_display = data["symbol"][i] if data["symbol"][i] else "-"

        # Add row with discrepancy highlighting (only for non-extended metrics)
        if has_discrepancy:
            # Highlight entire row with warning style
            table.add_row(
                data["date"][i].isoformat(),
                data["metric"][i],
                symbol_display,
                data["dataset"][i],
                value_analysis_display,
                value_db_display,
                value_final_display,
                error_display,
                data["tags"][i],
                style="bold yellow",
            )
        else:
            table.add_row(
                data["date"][i].isoformat(),
                data["metric"][i],
                symbol_display,
                data["dataset"][i],
                value_analysis_display,
                value_db_display,
                value_final_display,
                error_display,
                data["tags"][i],
            )

    # Print table
    console = Console()
    console.print(table)

    # Print summary of discrepancies using stats
    if stats.discrepancy_count > 0:
        console.print(
            f"\n[bold yellow]⚠️  Found {stats.discrepancy_count} row(s) with value discrepancies (excluding extended metrics)[/bold yellow]"
        )


def print_metrics_by_execution_id(metrics: Sequence[Metric], execution_id: str) -> None:
    """
    Display metrics for a specific execution in a formatted table.

    Shows all metrics from metrics_by_execution_id() in a table with columns:
    Date, Metric Name, Type, Dataset, Value, Tags

    Args:
        metrics: List of Metric objects from metrics_by_execution_id()
        execution_id: The execution ID to display in the title

    Example:
        >>> metrics = data.metrics_by_execution_id(db, execution_id)
        >>> print_metrics_by_execution_id(metrics, execution_id)
    """
    from rich.table import Table

    from dqx.data import metrics_to_pyarrow_table

    # Convert metrics to PyArrow table (handles sorting and formatting)
    pa_table = metrics_to_pyarrow_table(metrics, execution_id)

    # Create table with execution ID in title
    table = Table(title=f"Metrics for Execution: {execution_id}", show_lines=True)

    # Add columns with same color scheme as before
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Metric Name", style="yellow", no_wrap=True)
    table.add_column("Type")
    table.add_column("Dataset", style="magenta")
    table.add_column("Value")
    table.add_column("Tags", style="dim")

    # Convert PyArrow table to dict and iterate through rows
    data = pa_table.to_pydict()
    for i in range(pa_table.num_rows):
        # Format value with green color
        value_display = f"[green]{data['value'][i]}[/green]"

        # Add row (tags are already formatted by PyArrow function)
        table.add_row(
            data["date"][i].isoformat(),
            data["metric"][i],
            data["type"][i],
            data["dataset"][i],
            value_display,
            data["tags"][i],
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
    from rich.table import Table

    from dqx.data import symbols_to_pyarrow_table

    # Convert symbols to PyArrow table (handles ordering and formatting)
    pa_table = symbols_to_pyarrow_table(symbols)

    # Create table with title
    table = Table(title="Symbol Values", show_lines=True)

    # Add columns in specified order
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Symbol", style="yellow", no_wrap=True)
    table.add_column("Metric")
    table.add_column("Dataset", style="magenta")
    table.add_column("Value/Error")
    table.add_column("Tags", style="dim")

    # Convert PyArrow table to dict and iterate through rows
    data = pa_table.to_pydict()
    for i in range(pa_table.num_rows):
        # Combine Value and Error columns for display
        value = data["value"][i]
        error = data["error"][i]

        if value is not None:
            value_display = f"[green]{value}[/green]"
        else:
            value_display = f"[red]{error}[/red]"

        # Add row (tags are already formatted by PyArrow function)
        table.add_row(
            data["date"][i].isoformat(),
            data["symbol"][i],
            data["metric"][i],
            data["dataset"][i],
            value_display,
            data["tags"][i],
        )

    # Print table
    console = Console()
    console.print(table)


def print_analysis_report(report: AnalysisReport, symbol_lookup: dict[MetricKey, sp.Symbol]) -> None:
    """Display analysis reports in a formatted table.

    Args:
        report: AnalysisReport containing metrics from analysis
        symbol_lookup: Dictionary mapping metric keys to their symbolic representations
    """
    from rich.table import Table

    from dqx.data import analysis_reports_to_pyarrow_table

    # Convert reports to PyArrow table (handles sorting, symbol mapping, etc.)
    pa_table = analysis_reports_to_pyarrow_table(report, symbol_lookup)

    table = Table(title="Analysis Reports", show_lines=True)

    # Add columns with consistent color scheme
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Metric Name", style="yellow", no_wrap=True)
    table.add_column("Symbol", style="yellow", no_wrap=True)
    table.add_column("Type")
    table.add_column("Dataset", style="magenta")
    table.add_column("Value")
    table.add_column("Tags", style="dim")

    # Convert PyArrow table to dict and iterate through rows
    data = pa_table.to_pydict()
    for i in range(pa_table.num_rows):
        # Format value with green color
        value_display = f"[green]{data['value'][i]}[/green]"

        # Add row
        table.add_row(
            data["date"][i].isoformat(),
            data["metric"][i],
            data["symbol"][i],
            data["type"][i],
            data["dataset"][i],
            value_display,
            data["tags"][i],
        )

    # Print table
    console = Console()
    console.print(table)
