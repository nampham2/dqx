"""Display module for graph visualization using Rich."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence

import pyarrow as pa
import sympy as sp
from returns.result import Failure, Result, Success
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from dqx.common import MetricKey

if TYPE_CHECKING:
    from dqx.analyzer import AnalysisReport
    from dqx.common import AssertionResult, EvaluationFailure
    from dqx.graph.base import BaseNode
    from dqx.graph.traversal import Graph
    from dqx.models import Metric
    from dqx.orm.repositories import MetricDB
    from dqx.provider import SymbolInfo

# Type aliases for clarity
if TYPE_CHECKING:
    MetricValue = Result[float, list[EvaluationFailure]]
    SymbolValue = Result[float, str]


class NodeFormatter(Protocol):
    """Protocol for formatting tree nodes."""

    def format(self, node: BaseNode, tree_node: Tree) -> None:
        """Format a node and add to tree."""
        ...


def print_graph(graph: "Graph") -> None:
    """Print the dependency graph using Rich tree visualization."""
    console = Console()
    tree = Tree(f"[bold blue]{graph.root.name}[/bold blue]")

    for check in graph.root.children:
        check_branch = tree.add(f"[green]✓[/green] {check.name}")

        for assertion in check.children:
            # Format assertion with name and expression
            assertion_label = f"[yellow]→[/yellow] {assertion.name}: {assertion.actual}"
            if assertion.validator and assertion.validator.name:
                assertion_label += f" {assertion.validator.name}"
            check_branch.add(assertion_label)

    console.print(tree)


def print_assertion_results(results: list[AssertionResult]) -> None:
    """Display assertion results in a formatted table.

    Args:
        results: List of AssertionResult objects from collect_results()
    """
    console = Console()
    table = Table(show_header=True, header_style="bold cyan", title="Assertion Results")

    # Add columns
    table.add_column("Date", style="dim")
    table.add_column("Check", style="blue")
    table.add_column("Assertion", style="yellow")
    table.add_column("Expression", style="cyan")
    table.add_column("Severity", style="bright_magenta")
    table.add_column("Status", style="red")
    table.add_column("Value/Error", style="white")  # Combined column

    # Add rows
    for result in results:
        # Style for status column based on result
        if result.status == "FAILED":
            status_style = "[red]FAILED[/red]"
        elif result.status == "PASSED":
            status_style = "[green]PASSED[/green]"
        elif result.status == "SKIPPED":
            status_style = "[yellow]SKIPPED[/yellow]"
        else:
            status_style = "[dim]NOOP[/dim]"

        # Format value/error column
        if result.metric:
            match result.metric:
                case Success(value):
                    value_error_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                case Failure(errors):
                    # Format multiple error messages with bullets
                    if len(errors) > 1:
                        formatted_errors = "\n".join(f"• {err.error_message}" for err in errors)
                        value_error_str = f"[red]{formatted_errors}[/red]"
                    else:
                        value_error_str = f"[red]{errors[0].error_message}[/red]"
        else:
            value_error_str = "-"

        table.add_row(
            str(result.yyyy_mm_dd),
            result.check,
            result.assertion,
            result.expression if result.expression else "-",
            result.severity,
            status_style,
            value_error_str,
        )

    console.print(table)


def print_metrics_by_execution_id(metrics: Sequence[Metric], execution_id: str) -> None:
    """Display metrics from a specific execution in a formatted table.

    Args:
        metrics: Sequence of Metric objects from metrics_by_execution_id
        execution_id: The execution ID for these metrics
    """
    console = Console()

    # Convert to PyArrow table for consistent handling
    from dqx.data import metrics_to_pyarrow_table

    pa_table = metrics_to_pyarrow_table(metrics, execution_id)

    # Create Rich table with same columns as PyArrow table
    table = Table(show_header=True, header_style="bold cyan", title=f"Metrics for execution: {execution_id}")

    # Add columns
    table.add_column("Date", style="dim")
    table.add_column("Metric", style="green")
    table.add_column("Type", style="blue")
    table.add_column("Dataset", style="yellow")
    table.add_column("Value", style="white", justify="right")
    table.add_column("Tags", style="dim")

    # Convert PyArrow table to dict for easier access
    data = pa_table.to_pydict()

    # Add rows from PyArrow data
    for i in range(pa_table.num_rows):
        table.add_row(
            str(data["date"][i]),
            data["metric"][i],
            data["type"][i],
            data["dataset"][i],
            f"{data['value'][i]:.4f}",
            data["tags"][i],
        )

    console.print(table)
    console.print(f"\nTotal metrics: {len(metrics)}")


def print_symbols(symbols: list[SymbolInfo]) -> None:
    """Display all symbols in formatted table.

    Args:
        symbols: List of SymbolInfo objects from collect_symbols()
    """
    console = Console()

    # Convert to PyArrow table for consistent handling
    from dqx.data import symbols_to_pyarrow_table

    pa_table = symbols_to_pyarrow_table(symbols)

    # Create Rich table
    table = Table(show_header=True, header_style="bold cyan", title="Symbol Values")

    # Add columns
    table.add_column("Date", style="dim")
    table.add_column("Symbol", style="green")
    table.add_column("Metric", style="blue")
    table.add_column("Dataset", style="yellow")
    table.add_column("Value", style="white", justify="right")
    table.add_column("Error", style="red")
    table.add_column("Tags", style="dim")

    # Convert PyArrow table to dict for easier access
    data = pa_table.to_pydict()

    # Add rows from PyArrow data
    for i in range(pa_table.num_rows):
        value = data["value"][i]
        error = data["error"][i]

        # Format value
        value_str = f"{value:.4f}" if value is not None else "-"

        # Format error
        error_str = error if error is not None else ""

        table.add_row(
            str(data["date"][i]),
            data["symbol"][i],
            data["metric"][i],
            data["dataset"][i],
            value_str,
            error_str,
            data["tags"][i],
        )

    console.print(table)


def print_analysis_report(report: AnalysisReport, symbol_lookup: dict[MetricKey, sp.Symbol]) -> None:
    """Display analysis reports in a formatted table.

    Args:
        report: AnalysisReport containing metrics from analysis
        symbol_lookup: Dictionary mapping metric keys to their symbolic representations
    """
    from dqx.data import analysis_reports_to_pyarrow_table

    console = Console()

    # Convert reports to PyArrow table (handles sorting, symbol mapping, etc.)
    pa_table = analysis_reports_to_pyarrow_table(report, symbol_lookup)

    # Create Rich table
    table = Table(show_header=True, header_style="bold cyan", title="Analysis Report")

    # Add columns
    table.add_column("Date", style="dim")
    table.add_column("Metric", style="green")
    table.add_column("Symbol", style="blue")
    table.add_column("Type", style="cyan")
    table.add_column("Dataset", style="yellow")
    table.add_column("Value", style="white", justify="right")
    table.add_column("Tags", style="dim")

    # Convert PyArrow table to dict for easier access
    data = pa_table.to_pydict()

    # Add rows from PyArrow data
    for i in range(pa_table.num_rows):
        table.add_row(
            str(data["date"][i]),
            data["metric"][i],
            data["symbol"][i],
            data["type"][i],
            data["dataset"][i],
            f"{data['value'][i]:.4f}",
            data["tags"][i],
        )

    console.print(table)
    console.print(f"\nTotal metrics: {pa_table.num_rows}")


def _values_are_close(val1: Any, val2: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """Check if two values are close using epsilon comparison for numeric types.

    Args:
        val1: First value to compare
        val2: Second value to compare
        rel_tol: Relative tolerance for comparison
        abs_tol: Absolute tolerance for comparison

    Returns:
        True if values are close (or equal for non-numeric types)
    """
    # Import math here to avoid issues with auto-formatter
    import math

    # Handle None cases
    if val1 is None or val2 is None:
        return val1 == val2

    # Check if both values are numeric (int, float, or can be converted to float)
    try:
        # Try to convert to float to check if numeric
        float_val1 = float(val1)
        float_val2 = float(val2)
        # Use math.isclose for numeric comparison
        return math.isclose(float_val1, float_val2, rel_tol=rel_tol, abs_tol=abs_tol)
    except (TypeError, ValueError):
        # Non-numeric types, use direct equality
        return val1 == val2


def print_metric_trace(trace_table: pa.Table, data_av_threshold: float = 0.9) -> None:
    """Display metric trace table showing flow through the system.

    Args:
        trace_table: PyArrow table from metric_trace()
        data_av_threshold: Data availability threshold for color coding
    """
    console = Console()

    # Sort by symbol numeric index (x_1, x_2, ..., x_10, not x_1, x_10, x_2)
    import pyarrow.compute as pc

    # Extract numeric indices from symbol names for natural sorting
    symbol_column = trace_table.column("symbol")
    numeric_indices: list[float] = []
    for i in range(len(symbol_column)):
        symbol_name = str(symbol_column[i])
        if "_" in symbol_name:
            try:
                idx = int(symbol_name.split("_")[1])
                numeric_indices.append(float(idx))
            except (ValueError, IndexError):
                numeric_indices.append(float("inf"))  # Put non-standard symbols at end
        else:
            numeric_indices.append(float("inf"))

    # Create sort indices based on numeric values
    sort_indices = pc.sort_indices(pa.array(numeric_indices))
    trace_table = pc.take(trace_table, sort_indices)

    # Create Rich table (removed Tags column)
    table = Table(show_header=True, header_style="bold cyan", title="Metric Trace")

    # Add columns
    table.add_column("Date", style="dim")
    table.add_column("Metric", style="green")
    table.add_column("Symbol", style="blue")
    table.add_column("Dataset", style="yellow")
    table.add_column("Value DB", style="white", justify="right")
    table.add_column("Value Analysis", style="white", justify="right")
    table.add_column("Value/Error", style="white")  # Combined column for final value and error
    table.add_column("DAS", style="white", justify="right")  # Data Availability Score

    # Convert PyArrow table to dict for easier access
    data = trace_table.to_pydict()

    # Add rows from PyArrow data
    for i in range(trace_table.num_rows):
        # Format value columns
        value_db = data["value_db"][i]
        value_analysis = data["value_analysis"][i]
        value_final = data["value_final"][i]
        error = data["error"][i]
        is_extended = data["is_extended"][i] if "is_extended" in data else False

        # Format numeric values
        value_db_str = f"{value_db:.4f}" if value_db is not None else "-"
        value_analysis_str = f"{value_analysis:.4f}" if value_analysis is not None else "-"

        # Format combined value/error column
        if error is not None:
            value_error_str = f"[red]{error}[/red]"
        elif value_final is not None:
            value_error_str = f"{value_final:.4f}"
        else:
            value_error_str = "-"

        # Check for discrepancies and highlight using epsilon-based comparison
        has_discrepancy = False
        if not is_extended:  # Only check for non-extended metrics
            # Use epsilon-based comparison for numeric values
            if value_db is not None and value_analysis is not None and not _values_are_close(value_db, value_analysis):
                has_discrepancy = True
            if value_db is not None and value_final is not None and not _values_are_close(value_db, value_final):
                has_discrepancy = True
            if (
                value_analysis is not None
                and value_final is not None
                and not _values_are_close(value_analysis, value_final)
            ):
                has_discrepancy = True

        # Apply red styling to numeric values if discrepancy
        if has_discrepancy:
            value_db_str = f"[red]{value_db_str}[/red]"
            value_analysis_str = f"[red]{value_analysis_str}[/red]"
            if error is None and value_final is not None:  # Only color value, not error messages
                value_error_str = f"[red]{value_error_str}[/red]"

        # Format data availability with color coding
        data_av_ratio = data.get("data_av_ratio", [None] * trace_table.num_rows)[i]
        if data_av_ratio is not None:
            # Apply color coding based on threshold
            if data_av_ratio >= data_av_threshold:
                data_av_str = f"[green]{data_av_ratio:.2%}[/green]"
            elif data_av_ratio >= 0.5:
                data_av_str = f"[yellow]{data_av_ratio:.2%}[/yellow]"
            else:
                data_av_str = f"[red]{data_av_ratio:.2%}[/red]"
        else:
            data_av_str = "[dim]N/A[/dim]"

        table.add_row(
            str(data["date"][i]),
            data["metric"][i],
            data["symbol"][i],
            data["dataset"][i],
            value_db_str,
            value_analysis_str,
            value_error_str,
            data_av_str,
        )

    console.print(table)

    # Add explanation for DAS column
    console.print("\n[green]DAS[/green]: Data Availability Score")

    # Show discrepancy summary
    from dqx.data import metric_trace_stats

    stats = metric_trace_stats(trace_table)
    if stats.discrepancy_count > 0:
        console.print(
            f"\n[yellow]⚠️  Found {stats.discrepancy_count} discrepancies "
            f"(rows: {', '.join(str(r) for r in stats.discrepancy_rows[:5])}"
            f"{'...' if len(stats.discrepancy_rows) > 5 else ''})[/yellow]"
        )


def format_table_row(columns: list[tuple[str, str]], highlight: bool = False) -> str:
    """Format a row for table display.

    Args:
        columns: List of (value, style) tuples
        highlight: Whether to highlight the row

    Returns:
        Formatted row string
    """
    parts = []
    for value, style in columns:
        if highlight:
            parts.append(f"[bold {style}]{value}[/bold {style}]")
        else:
            parts.append(f"[{style}]{value}[/{style}]")
    return " | ".join(parts)


def display_metrics_by_execution_id(execution_id: str, db: "MetricDB") -> pa.Table:
    """Display metrics for a specific execution ID and return as PyArrow table.

    Args:
        execution_id: The execution ID to query
        db: The MetricDB instance

    Returns:
        PyArrow table containing the metrics
    """
    from dqx.data import metrics_to_pyarrow_table

    # Get metrics from database
    metrics = db.get_by_execution_id(execution_id)
    if not metrics:
        print(f"No metrics found for execution ID: {execution_id}")
        # Return empty table with correct schema
        return metrics_to_pyarrow_table([], execution_id)

    # Display in console
    print_metrics_by_execution_id(metrics, execution_id)

    # Return as PyArrow table
    return metrics_to_pyarrow_table(list(metrics), execution_id)
